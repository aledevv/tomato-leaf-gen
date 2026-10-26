"""Bend sampled from the G-series prior (PCA + Gaussian mixture fitted on Pheno4D leaflets).

The frozen resource (`utils/export_bend_prior.py`) holds a height field on the normalised leaf
domain: s in [0, 1] from the attachment to the tip, t in [-1, 1] from the left to the right
margin, as an S x T grid in units of the base-tip length. A bend is `mean + (z * scale) @
components` for a coefficient vector z drawn from the mixture. The deformer evaluates the field
at every mesh vertex through its semantic coordinates (longitudinal_u, margin_v), shifts it so
the attachment stays at height zero, and lifts the flat vertices along +z. The bend never depends
on the outline: the shape decides only where the field is sampled.

Two decoders can be frozen in the resource: the PCA (`mean`, `components`, `scale`, evaluated on
the grid) or a coordinate MLP (`dec_*` weights, Fourier features of (s, t) plus the code, evaluated
directly at every vertex). Two samplers: the Gaussian mixture or a flow-matching field (`flow_*`).
`temperature` scales the sampler's spread (mixture covariances by its square, the flow's starting
noise by it); below 1 it trims the tails a mixture fitted on few leaves overestimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from leafgen.bend.attachment import interpolate_margin_at_point
from leafgen.contracts.core import DeformationSample
from leafgen.contracts.geometry import LeafRole, SemanticFlatMesh2D


def mlp_forward(layers, x):
    """SiLU MLP on rows of x; `layers` = ((W, b), ...) with W of shape (out, in)."""
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = x / (1 + np.exp(-x))
    return x


@dataclass(frozen=True)
class BendPriorProvider:
    """Draw coefficient vectors from the frozen sampler (Gaussian mixture, or the flow-matching
    velocity field when the resource holds `flow_*` weights); roles share one prior."""

    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    grid: tuple[int, int]
    source: tuple[str, ...]
    flow: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    flow_steps: int = 0
    decoder: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    freqs: int = 0
    height_scale: float = 1.0
    temperature: float = 1.0

    @classmethod
    def load(cls, path: Path, temperature: float = 1.0) -> BendPriorProvider:
        with np.load(path, allow_pickle=False) as data:

            def layers(prefix):
                count = len([k for k in data.files if k.startswith(f"{prefix}_w")])
                return tuple(
                    (
                        np.array(data[f"{prefix}_w{i}"], dtype=np.float64),
                        np.array(data[f"{prefix}_b{i}"], dtype=np.float64),
                    )
                    for i in range(count)
                )

            flow, decoder = layers("flow"), layers("dec")
            provider = cls(
                flow=flow,
                flow_steps=int(data["flow_steps"]) if flow else 0,
                decoder=decoder,
                freqs=int(data["dec_freqs"]) if decoder else 0,
                height_scale=float(data["dec_height_scale"]) if decoder else 1.0,
                mean=np.array(data["mean"], dtype=np.float64) if "mean" in data else np.zeros(0),
                components=(
                    np.array(data["components"], dtype=np.float64)
                    if "components" in data
                    else np.zeros((0, 0))
                ),
                scale=np.array(data["scale"], dtype=np.float64) if "scale" in data else np.zeros(0),
                weights=np.array(data["weights"], dtype=np.float64),
                means=np.array(data["means"], dtype=np.float64),
                covariances=np.array(data["covariances"], dtype=np.float64),
                grid=(int(data["grid"][0]), int(data["grid"][1])),
                source=tuple(str(x) for x in data["source"]),
                temperature=float(temperature),
            )
        provider.validate()
        return provider

    @property
    def dims(self) -> int:
        return self.means.shape[1]

    def validate(self) -> None:
        k = self.dims
        if self.decoder:
            if (
                self.decoder[0][0].shape[1] != 2 + 4 * self.freqs + k
                or self.decoder[-1][0].shape[0] != 1
            ):
                raise ValueError("decoder weights do not match the Fourier features and the code")
        else:
            n = self.grid[0] * self.grid[1]
            if (
                self.components.shape != (k, n)
                or self.mean.shape != (n,)
                or self.scale.shape != (k,)
            ):
                raise ValueError("bend prior arrays are not aligned")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        c = len(self.weights)
        if self.means.shape != (c, k) or self.covariances.shape != (c, k, k):
            raise ValueError("bend prior mixture arrays are not aligned")
        if not np.isclose(self.weights.sum(), 1.0) or np.any(self.weights < 0):
            raise ValueError("mixture weights must be a distribution")
        if self.flow and (
            self.flow[0][0].shape[1] != k + 1
            or self.flow[-1][0].shape[0] != k
            or self.flow_steps < 1
        ):
            raise ValueError("flow weights do not match the number of components")

    def heights(self, coefficients: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Heights at normalised coordinates s in [0, 1], t in [-1, 1] for one code (MLP)."""
        z = np.asarray(coefficients, dtype=np.float64)
        f = np.arange(1, self.freqs + 1) * 2 * np.pi
        a, b = np.outer(s, f), np.outer(t, f)
        x = np.column_stack(
            [s, t, np.sin(a), np.cos(a), np.sin(b), np.cos(b), np.broadcast_to(z, (len(s), len(z)))]
        )
        return mlp_forward(self.decoder, x)[:, 0] * self.height_scale

    def field(self, coefficients: np.ndarray) -> np.ndarray:
        """Height field [S, T] of one coefficient vector (zeros give the mean bend)."""
        z = np.asarray(coefficients, dtype=np.float64)
        if self.decoder:
            s, t = np.meshgrid(
                np.linspace(0, 1, self.grid[0]), np.linspace(-1, 1, self.grid[1]), indexing="ij"
            )
            return self.heights(z, s.ravel(), t.ravel()).reshape(self.grid)
        return (self.mean + (z * self.scale) @ self.components).reshape(self.grid)

    def sample(self, *, seed: int, role: LeafRole) -> DeformationSample:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if role not in {"left", "right", "terminal"}:
            raise ValueError("role must be left, right or terminal")
        rng = np.random.Generator(np.random.PCG64(seed))
        component = -1
        if self.flow:  # Euler integration of the velocity field from Gaussian noise
            z = self.temperature * rng.standard_normal((1, self.dims))
            for i in range(self.flow_steps):
                t = np.full((1, 1), i / self.flow_steps)
                z = z + mlp_forward(self.flow, np.concatenate([z, t], 1)) / self.flow_steps
            z = z[0]
        else:
            component = int(rng.choice(len(self.weights), p=self.weights))
            z = rng.multivariate_normal(
                self.means[component], self.covariances[component] * self.temperature**2
            )
        return DeformationSample(
            parameters=z,
            requested_role=role,
            source_role="terminal" if role == "terminal" else "lateral",
            source_id=f"{self.source[0]}/seed{seed}",
            provenance={
                "model": self.source[0],
                "training_data_sha256": self.source[1],
                "training": self.source[2],
                "sampler": "flow" if self.flow else "gmm",
                "mixture_component": component,
                "temperature": self.temperature,
                "components": int(len(z)),
                "role_specific": False,
            },
        )


@dataclass(frozen=True)
class BendPriorDeformer:
    """Lift the flat semantic mesh by the prior's height field (attachment stays at z = 0)."""

    attachment_tolerance: float = 1e-10
    provider: BendPriorProvider | None = None

    def lift(
        self, mesh: SemanticFlatMesh2D, evaluate
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Lift the vertices by `evaluate(s, t)` (heights at normalised coordinates), with the
        attachment at height zero; returns vertices, attachment, tip, attachment_v."""
        attachment_v = interpolate_margin_at_point(
            mesh, np.asarray([0.0, 0.0]), tolerance=self.attachment_tolerance
        )
        s = np.clip(mesh.longitudinal_u, 0.0, 1.0)
        t = np.clip(mesh.margin_v, -1.0, 1.0)
        at_attachment, at_tip = evaluate(np.array([0.0, 1.0]), np.array([attachment_v, 0.0]))
        height = evaluate(s, t) - at_attachment
        vertices = np.column_stack([mesh.mesh.vertices_xy, height])
        tip = np.asarray([0.0, 1.0, float(at_tip - at_attachment)])
        return vertices, np.zeros(3, dtype=np.float64), tip, attachment_v

    def apply_field(
        self, mesh: SemanticFlatMesh2D, field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        rows, cols = field.shape
        return self.lift(
            mesh,
            lambda s, t: map_coordinates(
                field, [s * (rows - 1), (t + 1.0) / 2.0 * (cols - 1)], order=1, mode="nearest"
            ),
        )

    def apply(
        self, mesh: SemanticFlatMesh2D, sample: DeformationSample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.provider is None:
            raise ValueError("BendPriorDeformer needs the provider that decodes coefficients")
        z = sample.parameters
        if self.provider.decoder:
            return self.lift(mesh, lambda s, t: self.provider.heights(z, s, t))
        return self.apply_field(mesh, self.provider.field(z))
