"""Shape model "rae": regularised autoencoder on the ordered points, Gaussian mixture
fitted ex post on the codes.

Ghosh, Sajjadi, Vergari, Black & Schölkopf, "From Variational to Deterministic Autoencoders"
(ICLR 2020): a deterministic encoder/decoder trained for reconstruction with an L2 penalty on the
codes and weight decay on the decoder (RAE-L2), then an "ex-post" density on the training codes
(here a full-covariance Gaussian mixture, size by BIC, one per subset). Encoder and decoder are
1-D convolutions with circular padding on the 2 x 256 pseudo-landmarks (base at index 0, tip at
128); the reconstruction loss is the point-wise L2 plus the L2 of the discrete curvature (second
differences) so that the teeth survive. Trained once on the pooled subsets (``fit`` takes a list
of banks), validated on 1-NNA / Z_U of draws from a provisional 5-component mixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from leafgen.contracts.geometry import LeafRole, LeafShape2D
from leafgen.shape.features import ROLES, efd_features
from leafgen.shape.models import neural
from leafgen.shape.models.gmm import _fit_gmm

NAME = "rae"
LATENT, CURVATURE, CODE_L2 = 32, 1.0, 1e-3


def build(latent: int, widths=(32, 64, 128, 128)):
    import torch
    from torch import nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv1d(cin, cout, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            enc, cin, length = [], 2, 256
            for w in widths:
                enc += [block(cin, w), nn.AvgPool1d(2)]
                cin, length = w, length // 2
            self.encoder = nn.Sequential(*enc, nn.Flatten(), nn.Linear(cin * length, latent))
            self.length0, self.c0 = length, cin
            self.expand = nn.Linear(latent, cin * length)
            dec = []
            for w in reversed(widths):
                dec += [nn.Upsample(scale_factor=2, mode="linear"), block(cin, w)]
                cin = w
            self.decoder = nn.Sequential(
                *dec, nn.Conv1d(cin, 2, 3, padding=1, padding_mode="circular")
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            h = self.expand(z).reshape(len(z), self.c0, self.length0)
            return self.decoder(h)

        def forward(self, x):
            return self.decode(self.encode(x))

    return AE(), torch


def _loss(model, x, subset, vector, generator):  # noqa: ARG001 (neural.train signature)
    z = model.encode(x)
    y = model.decode(z)
    curvature = lambda a: a - 0.5 * (a.roll(1, dims=-1) + a.roll(-1, dims=-1))  # noqa: E731
    return (
        ((y - x) ** 2).mean()
        + CURVATURE * ((curvature(y) - curvature(x)) ** 2).mean() * 64
        + CODE_L2 * (z**2).sum(1).mean()
    )


def fit(
    bank_path: Path | list[Path],
    output: Path,
    *,
    latent: int = LATENT,
    minutes: float = 40.0,
    dataset: str = "",
    seed: int = 0,
    **_: Any,
) -> dict[str, Any]:
    from sklearn.mixture import GaussianMixture

    banks = [Path(b) for b in (bank_path if isinstance(bank_path, list) else [bank_path])]
    data = neural.pooled(banks)
    val = neural.split_val(data["photo"], seed=seed)
    mean, std = data["landmarks"][~val].mean(0), float(data["landmarks"][~val].std())
    model, torch = build(latent)
    dev = neural.device()
    model.to(dev)
    x = torch.as_tensor(
        ((data["landmarks"] - mean) / std).reshape(-1, 256, 2).transpose(0, 2, 1),
        dtype=torch.float32,
        device=dev,
    )
    train_idx, val_idx = np.flatnonzero(~val), np.flatnonzero(val)
    subset = torch.as_tensor(data["subset"], device=dev)

    def decode_np(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = model.decode(torch.as_tensor(z, dtype=torch.float32, device=dev)).cpu().numpy()
        return y.transpose(0, 2, 1).reshape(len(z), -1) * std + mean

    def codes(indices, net) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate(
                [
                    net.encode(x[indices[i : i + 2048]]).cpu().numpy()
                    for i in range(0, len(indices), 2048)
                ]
            )

    def validate(average):
        c = codes(train_idx, average)
        rng = np.random.default_rng(seed + 1)
        gmm = GaussianMixture(5, covariance_type="full", random_state=seed).fit(
            c[rng.choice(len(c), min(5000, len(c)), replace=False)]
        )
        z = gmm.sample(neural.VAL_SAMPLES)[0]
        with torch.no_grad():
            y = average.decode(torch.as_tensor(z, dtype=torch.float32, device=dev)).cpu().numpy()
        gen = y.transpose(0, 2, 1).reshape(len(z), -1) * std + mean
        with torch.no_grad():
            rec = float(((average(x[val_idx[:1000]]) - x[val_idx[:1000]]) ** 2).mean())
        return dict(
            **neural.landmarks_metrics(
                gen, data["landmarks"][val_idx], data["landmarks"][train_idx]
            ),
            rec_mse=rec,
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "train.log").open("w")
    run = neural.train(
        model,
        x[train_idx],
        subset[train_idx],
        loss_fn=_loss,
        minutes=minutes,
        seed=seed,
        validate=validate,
        lr=1e-3,
        log=lambda s: (print(s), log.write(s + "\n"), log.flush()),
    )
    log.close()
    torch.save(model.state_dict(), output / "model.pt")
    np.savez(output / "norm.npz", mean=mean, std=np.array([std]))
    density, files = {}, ["model.pt", "norm.npz"]
    all_codes = codes(train_idx, model)
    for name in data["names"]:
        index = neural.SUBSETS.index(name) if name in neural.SUBSETS else None
        selected = all_codes[data["subset"][train_idx] == index]
        arrays, info = _fit_gmm(selected, 0, seed)
        np.savez(output / f"density_{name}.npz", **arrays)
        density[name] = dict(count=int(len(selected)), **info)
        files.append(f"density_{name}.npz")
    label = dataset or "pool"
    metadata = dict(
        model=NAME,
        representation="landmarks_128",
        latent=latent,
        subsets=data["names"],
        roles=list(ROLES),
        role_specific=False,
        dataset=label,
        version=f"{label.lower()}_rae_z{latent}",
        banks=[str(b) for b in banks],
        training_counts={r: int(len(train_idx)) for r in ROLES},
        validation_leaflets=int(val.sum()),
        training=dict(steps=run["steps"], minutes=run["minutes"], history=run["history"]),
        density=density,
        sha256={f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    Generator(output, subset=data["names"][0])
    return metadata


class Generator:
    """Draw codes from the subset mixture and decode them (``subset``: first fitted by default)."""

    def __init__(
        self,
        model: Path,
        *,
        subset: str | None = None,
        maximum_geometry_attempts: int = 100,
        **_: Any,
    ) -> None:
        model = Path(model)
        self.maximum_geometry_attempts = maximum_geometry_attempts
        self.metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
        for name, digest in self.metadata.get("sha256", {}).items():
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{NAME} model checksum mismatch: {name}")
        self.subset = subset or self.metadata["subsets"][0]
        if self.subset not in self.metadata["subsets"]:
            raise ValueError(f"{NAME}: no density for subset {self.subset}")
        self.net, self.torch = build(self.metadata["latent"])
        self.net.load_state_dict(self.torch.load(model / "model.pt", map_location="cpu"))
        self.device = neural.device()
        self.net.to(self.device).eval()
        with np.load(model / "norm.npz", allow_pickle=False) as d:
            self.mean, self.std = d["mean"], float(d["std"][0])
        with np.load(model / f"density_{self.subset}.npz", allow_pickle=False) as d:
            self.density = {k: d[k] for k in d.files}

    def sample(self, count: int, seed: int) -> list[np.ndarray]:
        rng = np.random.Generator(np.random.PCG64(seed))
        p = self.density
        out: list[np.ndarray] = []
        attempts = 0
        while len(out) < count and attempts < self.maximum_geometry_attempts:
            n = min(2048, max(64, count - len(out)))
            k = rng.choice(len(p["weights"]), size=n, p=p["weights"])
            z = np.array([rng.multivariate_normal(p["means"][i], p["covariances"][i]) for i in k])
            with self.torch.no_grad():
                y = (
                    self.net.decode(
                        self.torch.as_tensor(z, dtype=self.torch.float32, device=self.device)
                    )
                    .cpu()
                    .numpy()
                )
            landmarks = y.transpose(0, 2, 1).reshape(n, -1) * self.std + self.mean
            out += [c for c in neural.contours_from_landmarks(landmarks) if c is not None]
            attempts += 1
        if len(out) < count:
            raise RuntimeError(f"{NAME}: only {len(out)} valid outlines of {count}")
        return out[:count]

    def generate(self, *, seed: int, role: LeafRole) -> LeafShape2D:
        if role not in ROLES:
            raise ValueError("role must be left, right or terminal")
        points = self.sample(1, seed)[0]
        return LeafShape2D(
            contour_xy=points,
            biological_base_xy=np.array([0.0, 0.0]),
            biological_tip_xy=np.array([0.0, 1.0]),
            features_h60=efd_features(points),
            provenance={
                "generator": NAME,
                "provider": f"{NAME}_z{self.metadata['latent']}",
                "model_version": self.metadata.get("version"),
                "model_sha256": dict(self.metadata.get("sha256", {})),
                "dataset": self.metadata.get("dataset"),
                "subset": self.subset,
                "role": role,
                "seed": seed,
                "reflection_applied": False,
            },
        )
