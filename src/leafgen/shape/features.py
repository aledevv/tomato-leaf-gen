"""Stage 5: outline features and the shape bank.

Representation: elliptic Fourier descriptors with 60 harmonics (EFD H60) of the canonical contour
(base (0,0), tip (0,1), 1024 points, counter-clockwise). ``efd_features`` goes from contour to the
242-vector ``[dc_x, dc_y, (a, b, c, d) x 60]``, ``reconstruct_h60`` goes back, ``valid_contour``
says whether a reconstructed contour is usable (finite, positive area, no self-intersection).
``build_bank`` turns a ``canonical/`` folder into ``bank.npz`` = features of every leaflet whose H60
reconstruction is valid (the few that self-intersect are listed in the manifest) plus ids, split,
role and a PCA for the k-NN index of the i3 model. To use another representation,
replace these functions and keep the bank layout (``features`` is ``(N, D)``).

    python -m leafgen.shape.features --canonical W_prep/canonical --output bank.npz
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from leafgen.contracts.geometry import LeafRole, signed_area
from leafgen.jsonl import read_jsonl

HARMONICS = 60
FEATURE_DIM = 2 + 4 * HARMONICS  # 242
POINT_COUNT = 1024
ROLES: tuple[LeafRole, ...] = ("left", "right", "terminal")
BANK_ROLES = (*ROLES, "generic")


# ----------------------------------------------------------------------------- contour -> features
def efd_features(points: np.ndarray, harmonics: int = HARMONICS) -> np.ndarray:
    """EFD of a closed CCW contour, start point anchored at the lowest (|p|², y, x) vertex."""
    contour = np.asarray(points, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1:] != (2,) or signed_area(contour) <= 0.0:
        raise ValueError("EFD input must be a CCW Nx2 contour")
    distance2 = np.einsum("ij,ij->i", contour, contour)
    anchor = int(np.lexsort((contour[:, 0], contour[:, 1], distance2))[0])
    contour = np.roll(contour, -anchor, axis=0)
    following = np.roll(contour, -1, axis=0)
    delta = following - contour
    dt = np.linalg.norm(delta, axis=1)
    keep = dt > 1e-12
    contour = contour[keep]
    following = np.roll(contour, -1, axis=0)
    delta = following - contour
    dt = np.linalg.norm(delta, axis=1)
    perimeter = float(dt.sum())
    if len(contour) < 4 or perimeter <= 1e-12:
        raise ValueError("EFD input contour is degenerate")
    t_end = np.cumsum(dt)
    t_start = np.concatenate([[0.0], t_end[:-1]])
    velocity = delta / dt[:, None]
    dc = np.sum(dt[:, None] * (contour + following) * 0.5, axis=0) / perimeter
    coefficients = np.empty((harmonics, 4), dtype=np.float64)
    for index in range(harmonics):
        n = index + 1
        omega = 2.0 * math.pi * n / perimeter
        cos_delta = np.cos(omega * t_end) - np.cos(omega * t_start)
        sin_delta = np.sin(omega * t_end) - np.sin(omega * t_start)
        factor = perimeter / (2.0 * math.pi**2 * n**2)
        coefficients[index] = [
            factor * np.sum(velocity[:, 0] * cos_delta),
            factor * np.sum(velocity[:, 0] * sin_delta),
            factor * np.sum(velocity[:, 1] * cos_delta),
            factor * np.sum(velocity[:, 1] * sin_delta),
        ]
    return np.concatenate([dc, coefficients.ravel()])


# ----------------------------------------------------------------------------- features -> contour
@lru_cache(maxsize=4)
def _h60_basis(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    phase = np.arange(point_count, dtype=np.float64) / point_count
    angles = 2.0 * np.pi * phase[:, None] * np.arange(1, HARMONICS + 1, dtype=np.float64)[None, :]
    cosine, sine = np.cos(angles), np.sin(angles)
    cosine.setflags(write=False)
    sine.setflags(write=False)
    return cosine, sine


def reconstruct_h60(features: np.ndarray, *, point_count: int = POINT_COUNT) -> np.ndarray:
    """Closed contour (point_count x 2) from one 242-vector."""
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"H60 reconstruction requires one finite {FEATURE_DIM}-vector")
    coefficients = vector[2:].reshape(HARMONICS, 4)
    cosine, sine = _h60_basis(point_count)
    return np.column_stack(
        [
            vector[0] + cosine @ coefficients[:, 0] + sine @ coefficients[:, 1],
            vector[1] + cosine @ coefficients[:, 2] + sine @ coefficients[:, 3],
        ]
    )


def _proper_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab, cd = b - a, d - c
    cross1 = ab[0] * (c[1] - a[1]) - ab[1] * (c[0] - a[0])
    cross2 = ab[0] * (d[1] - a[1]) - ab[1] * (d[0] - a[0])
    cross3 = cd[0] * (a[1] - c[1]) - cd[1] * (a[0] - c[0])
    cross4 = cd[0] * (b[1] - c[1]) - cd[1] * (b[0] - c[0])
    return bool(cross1 * cross2 < -1e-12 and cross3 * cross4 < -1e-12)


def _self_intersects(points: np.ndarray) -> bool:
    """Segment pairs bucketed on a grid, so the check is far from quadratic."""
    ends = np.roll(points, -1, axis=0)
    minimum = np.minimum(points.min(axis=0), ends.min(axis=0))
    extent = np.maximum(np.maximum(points.max(axis=0), ends.max(axis=0)) - minimum, 1e-12)
    grid_size = max(8, min(64, int(round(math.sqrt(len(points)) * 2))))
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (start, end) in enumerate(zip(points, ends, strict=True)):
        low = np.floor((np.minimum(start, end) - minimum) / extent * grid_size).astype(int)
        high = np.floor((np.maximum(start, end) - minimum) / extent * grid_size).astype(int)
        low, high = np.clip(low, 0, grid_size - 1), np.clip(high, 0, grid_size - 1)
        for x_index in range(int(low[0]), int(high[0]) + 1):
            for y_index in range(int(low[1]), int(high[1]) + 1):
                buckets.setdefault((x_index, y_index), []).append(index)
    checked: set[tuple[int, int]] = set()
    count = len(points)
    for indices in buckets.values():
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                pair = (min(first, second), max(first, second))
                if pair in checked:
                    continue
                checked.add(pair)
                if second in {(first - 1) % count, first, (first + 1) % count}:
                    continue
                if _proper_intersection(points[first], ends[first], points[second], ends[second]):
                    return True
    return False


def valid_contour(points: np.ndarray) -> bool:
    contour = np.asarray(points, dtype=np.float64)
    return bool(
        contour.shape == (POINT_COUNT, 2)
        and np.all(np.isfinite(contour))
        and signed_area(contour) > 1e-12
        and not _self_intersects(contour)
    )


# ----------------------------------------------------------------------------- pseudo-landmarks
LANDMARKS_PER_SIDE = 128
LANDMARK_DIM = 4 * LANDMARKS_PER_SIDE  # 512


def _resample_open(points: np.ndarray, count: int) -> np.ndarray:
    """``count`` points by arc length along an open polyline, first point kept, last excluded."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.arange(count) * cumulative[-1] / count
    return np.column_stack([np.interp(targets, cumulative, points[:, j]) for j in (0, 1)])


def pseudo_landmarks(contour: np.ndarray, per_side: int = LANDMARKS_PER_SIDE) -> np.ndarray:
    """Semilandmarks of a canonical CCW contour: ``per_side`` points by arc length from the base
    (vertex nearest (0, 0)) to the tip (vertex nearest (0, 1)), then ``per_side`` from the tip back
    to the base; flattened ``[x0, y0, x1, y1, ...]`` (4 * per_side values). The canonical frame
    already fixes base and tip, so no Procrustes step."""
    pts = np.asarray(contour, dtype=np.float64)
    pts = np.roll(pts, -int(np.argmin(np.einsum("ij,ij->i", pts, pts))), axis=0)
    tip = int(np.argmin(np.linalg.norm(pts - [0.0, 1.0], axis=1)))
    if tip < 2 or tip > len(pts) - 3:
        raise ValueError("tip and base coincide on the contour")
    first = _resample_open(pts[: tip + 1], per_side)
    second = _resample_open(np.vstack([pts[tip:], pts[:1]]), per_side)
    return np.vstack([first, second]).ravel()


def landmarks_to_contour(vector: np.ndarray, *, point_count: int = POINT_COUNT) -> np.ndarray:
    """Closed contour (point_count x 2) through the landmarks, resampled by arc length."""
    from leafgen.shape.canonicalize import resample_closed

    return resample_closed(np.asarray(vector, dtype=np.float64).reshape(-1, 2), point_count)


# ----------------------------------------------------------------------------- PCA and the bank
def pca(
    features: np.ndarray, dimension: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, components (sign fixed: largest entry positive) and singular values, by SVD.
    ``dimension`` None keeps the components covering 95 % of the variance, clamped to 5–32."""
    mean = features.mean(axis=0)
    _, singular, components = np.linalg.svd(features - mean, full_matrices=False)
    if dimension is None:
        variance = singular**2
        cumulative = np.cumsum(variance) / variance.sum()
        dimension = max(5, min(int(np.searchsorted(cumulative, 0.95) + 1), 32))
    selected = components[:dimension].copy()
    for row in selected:
        if row[int(np.argmax(np.abs(row)))] < 0.0:
            row *= -1.0
    return mean, selected, singular


@dataclass(frozen=True)
class ShapeBank:
    """Every leaflet's features plus ids, split, role and a PCA (used by the i3 neighbour index)."""

    instance_id: np.ndarray
    source_path: np.ndarray
    split: np.ndarray
    role: np.ndarray
    features: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray

    FIELDS = (
        "instance_id",
        "source_path",
        "split",
        "role",
        "features",
        "pca_mean",
        "pca_components",
    )

    @classmethod
    def load(cls, path: Path) -> ShapeBank:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(cls.FIELDS):
                raise ValueError(f"unexpected shape-bank fields: {sorted(data.files)}")
            values = {name: np.array(data[name], copy=True) for name in cls.FIELDS}
        bank = cls(**values)
        bank.validate()
        return bank

    def validate(self) -> None:
        count = len(self.instance_id)
        if count < 9 or len(set(map(str, self.instance_id))) != count:
            raise ValueError("shape bank needs at least nine unique entries")
        if any(len(value) != count for value in (self.source_path, self.split, self.role)):
            raise ValueError("shape-bank metadata arrays are not aligned")
        if self.features.ndim != 2 or len(self.features) != count:
            raise ValueError("shape-bank features must be (N, D)")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("shape bank requires finite feature vectors")
        dimension = self.features.shape[1]
        if self.pca_mean.shape != (dimension,) or self.pca_components.shape[1:] != (dimension,):
            raise ValueError("shape-bank PCA arrays do not match the features")
        if np.any(self.split == "holdout"):
            raise ValueError("holdout entries are forbidden in the shape bank")
        roles = set(map(str, np.unique(self.role)))
        if not roles.issubset(BANK_ROLES):
            raise ValueError("shape bank contains an unknown role")
        if "generic" in roles and roles != {"generic"}:
            raise ValueError("generic and role-conditioned shape entries cannot be mixed")

    @property
    def scores(self) -> np.ndarray:
        return (self.features - self.pca_mean) @ self.pca_components.T

    def development(self) -> tuple[np.ndarray, np.ndarray]:
        """Features and roles of the development split (what models are fitted on)."""
        selected = self.split == "development"
        return self.features[selected], self.role[selected].astype(str)


def build_bank(
    *,
    canonical: Path,
    output: Path,
    include_holdout: bool = False,
    holdout_fraction: float = 0.0,
    seed: int = 0,
    avoid: Path | None = None,
) -> dict[str, Any]:
    """EFD features of every contour of ``canonical/catalog.jsonl`` -> ``output`` (.npz).
    ``holdout_fraction`` > 0 sets that share of the photos aside (by photo, deterministic) and
    writes them to ``<output>_holdout.npz`` for ``evaluate --reference``. The photos of the bank
    ``avoid`` (an earlier held-out split, already used to compare models) are never set aside."""
    canonical, output = Path(canonical).resolve(), Path(output).resolve()
    rows = read_jsonl(canonical / "catalog.jsonl")
    if holdout_fraction > 0:
        photos = sorted({row["source_group"] for row in rows})
        used = set(map(str, ShapeBank.load(Path(avoid)).source_path)) if avoid else set()
        candidates = [p for p in photos if p not in used]
        rng = np.random.default_rng(seed)
        held = set(rng.choice(candidates, int(round(holdout_fraction * len(photos))), replace=False))
        held_rows = [dict(row, split="development") for row in rows if row["source_group"] in held]
        rows = [row for row in rows if row["source_group"] not in held]
        _write_bank(canonical, held_rows, output.with_name(output.stem + "_holdout.npz"))
    return _write_bank(canonical, rows, output, include_holdout)


def _write_bank(
    canonical: Path, rows: list[dict], output: Path, include_holdout: bool = False
) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if include_holdout or row["split"] != "holdout"),
        key=lambda row: row["leaflet_id"],
    )
    if not selected:
        raise ValueError("canonical catalog has no usable shapes")
    features, dropped = [], []
    for row in list(selected):
        with np.load(canonical / row["contour"], allow_pickle=False) as data:
            vector = efd_features(data["contour_xy"])
        if valid_contour(reconstruct_h60(vector)):  # a few margins self-intersect once truncated
            features.append(vector)
        else:
            dropped.append(row["leaflet_id"])
            selected.remove(row)
    matrix = np.asarray(features, dtype=np.float64)
    mean, components, _ = pca(matrix)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        instance_id=np.asarray([row["leaflet_id"] for row in selected]),
        source_path=np.asarray([row["source_group"] for row in selected]),
        split=np.asarray([row["split"] for row in selected]),
        role=np.asarray([row["role"] for row in selected]),
        features=matrix,
        pca_mean=mean,
        pca_components=components,
    )
    summary = validate_bank(output)
    manifest = {
        "bank": output.name,
        "representation": "efd_h60",
        "harmonics": HARMONICS,
        "pca_components": int(len(components)),
        "dropped_invalid_h60": dropped,
        **summary,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_bank(path: Path) -> dict[str, Any]:
    """Load the bank and check that every feature vector reconstructs a usable contour."""
    bank = ShapeBank.load(path)
    invalid = sum(not valid_contour(reconstruct_h60(f)) for f in bank.features)
    if invalid:
        raise ValueError(f"shape bank contains {invalid} invalid H60 reconstructions")
    return {
        "entries": int(len(bank.instance_id)),
        "roles": {
            r: int(np.count_nonzero(bank.role == r)) for r in BANK_ROLES if np.any(bank.role == r)
        },
        "source_groups": len(set(map(str, bank.source_path))),
        "development_entries": int(np.count_nonzero(bank.split == "development")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--canonical", type=Path, required=True, help="canonical/ folder")
    parser.add_argument("--output", type=Path, required=True, help="bank .npz to write")
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="share of photos set aside in <output>_holdout.npz",
    )
    parser.add_argument("--seed", type=int, default=0, help="draw of the held-out photos")
    parser.add_argument(
        "--avoid", type=Path, default=None, help="bank whose photos are never held out"
    )
    args = parser.parse_args()
    manifest = build_bank(
        canonical=args.canonical,
        output=args.output,
        holdout_fraction=args.holdout,
        seed=args.seed,
        avoid=args.avoid,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
