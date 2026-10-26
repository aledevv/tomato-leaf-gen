"""Stage 7: how close are the outlines a shape model draws to a set of real outlines?

    python -m leafgen.shape.evaluate --model MODEL --reference HOLDOUT.npz \
        [--train BANK.npz] [--count N] [--label name] [--seed 0]

``MODEL`` is a fitted model folder (``metadata.json`` names the model: rae_flow, mask_diffusion)
or ``train:<bank.npz>`` (real development leaflets drawn at random, the ceiling a generator can
reach). ``--reference`` holds the real contours: a holdout bank (``features --holdout``) or a
``canonical/`` folder (``catalog.jsonl`` + ``contours/*.npz``). Every row should draw as many
samples as the reference holds (``--count`` defaults to that).

Three views, all with published definitions:

- set metrics in two spaces, **pseudo-landmarks** (``features.pseudo_landmarks``: 2 x 128 points
  by arc length anchored at base and tip, Euclidean distance; the space that sees the margin and
  belongs to no model) and **EFD-PCA-16** (PCA fitted on the real EFD features, the earlier view):
  density / coverage (Naeem et al. 2020, k = 5) and 1-NNA (Lopez-Paz & Oquab 2017), with the
  subsampling interval and the real-vs-real reference of ``leafgen.metrics.distribution``;
- data copying (Meehan, Chaudhuri & Dasgupta, AISTATS 2020), in the landmark space, when
  ``--train`` gives the bank the model was fitted on: Z_U compares the distances to the training
  set of the generated and of the held-out leaflets (Z_U << 0 copying, >> 0 underfitting), C_T
  averages Z_U over k-means cells of the training set; the reference is one half of the held-out
  against the other (ideal ~0). Also the median distance to the closest training leaflet and the
  share of draws whose nearest/second-nearest training ratio is below 1/3 (Yoon et al. 2023);
- descriptors: solidity, width/length, circularity of generated vs real outlines, Wasserstein-1
  distance and the two medians.

Writes ``results/shape/<label>.json`` and ``.md``; ``--table`` collects every json of that
folder into ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from leafgen.jsonl import read_jsonl
from leafgen.paths import ROOT
from leafgen.shape.features import (
    ShapeBank,
    efd_features,
    pca,
    pseudo_landmarks,
    reconstruct_h60,
)
from leafgen.shape.models import MODELS

OUT = ROOT / "results/shape"
ROLES = ("left", "right", "terminal")


class TrainingShapes:
    """Baseline: a real development leaflet of the bank, drawn at random by the seed."""

    def __init__(self, bank: Path) -> None:
        features, _ = ShapeBank.load(bank).development()
        self.features = features

    def sample(self, count: int, seed: int) -> list[np.ndarray]:
        """``count`` distinct leaflets (no duplicates: two identical draws would be each other's
        nearest neighbour and bias 1-NNA)."""
        count = min(count, len(self.features))  # all of them when the reference is larger
        indices = np.random.default_rng(seed).choice(len(self.features), count, replace=False)
        return [reconstruct_h60(self.features[i]) for i in indices]


def load_generator(model: str, subset: str | None = None, steps: int | None = None):
    text = str(model)
    if text.startswith("train:"):
        return TrainingShapes(Path(text[6:])), "real training leaflets"
    path = Path(text)
    name = json.loads((path / "metadata.json").read_text())["model"]
    options = {k: v for k, v in (("subset", subset), ("steps", steps)) if v}
    return MODELS[name].Generator(path, **options), name


def descriptors(contour: np.ndarray) -> tuple[float, float, float]:
    """solidity, width/length, circularity of a canonical contour."""
    pts = np.round(contour * 2000).astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(pts)
    hull = cv2.contourArea(cv2.convexHull(pts)) or 1.0
    perimeter = cv2.arcLength(pts, True) or 1.0
    return (
        area / hull,
        float(np.ptp(contour[:, 0]) / np.ptp(contour[:, 1])),
        4 * np.pi * area / perimeter**2,
    )


def set_metrics(real: np.ndarray, generated: np.ndarray) -> dict:
    """Density / coverage / 1-NNA, their subsampling interval and the real-vs-real reference,
    Euclidean distances between the rows."""
    from leafgen.metrics.distribution import (
        interval,
        metrics,
        real_vs_real,
        rescaled_interval,
        subsamples,
    )

    rr, gg, rg = cdist(real, real), cdist(generated, generated), cdist(real, generated)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)
    runs = subsamples(rr, gg, rg)  # the same subsamples for both intervals
    return dict(
        metrics=metrics(rr, gg, rg),
        interval=interval(rr, gg, rg, runs=runs),
        interval_95=rescaled_interval(rr, gg, rg, runs=runs),
        real_vs_real=real_vs_real(rr),
    )


def load_reference(reference: Path) -> list[np.ndarray]:
    if reference.suffix == ".npz":  # a (holdout) bank: real contours from their EFD vectors
        return [reconstruct_h60(f) for f in ShapeBank.load(reference).features]
    rows = read_jsonl(reference / "catalog.jsonl")
    return [np.load(reference / r["contour"])["contour_xy"] for r in rows]


def evaluate(
    model: str,
    reference: Path,
    count: int | None,
    seed: int,
    train: Path | None = None,
    subset: str | None = None,
    steps: int | None = None,
) -> dict:
    from leafgen.metrics.distribution import data_copying

    real = load_reference(reference)
    count = count or len(real)
    generator, kind = load_generator(model, subset, steps)
    if hasattr(generator, "sample"):
        generated = generator.sample(count, seed)
    else:
        generated = [
            generator.generate(seed=seed + i, role=ROLES[i % 3]).contour_xy for i in range(count)
        ]
    real_l = np.array([pseudo_landmarks(c) for c in real])
    gen_l = np.array([pseudo_landmarks(c) for c in generated])
    real_f = np.array([efd_features(c) for c in real])
    gen_f = np.array([efd_features(c) for c in generated])
    mean, components, _ = pca(real_f, 16)
    result = dict(
        model=str(model),
        kind=kind,
        steps=steps,
        reference=str(reference),
        train=str(train) if train else None,
        real=len(real),
        generated=len(generated),
        landmark=set_metrics(real_l, gen_l),
        efd_pca16=set_metrics((real_f - mean) @ components.T, (gen_f - mean) @ components.T),
    )
    if train is not None:
        train_l = np.array(
            [pseudo_landmarks(reconstruct_h60(f)) for f in ShapeBank.load(train).development()[0]]
        )
        order = np.random.default_rng(seed).permutation(len(real_l))
        a, b = order[: len(order) // 2], order[len(order) // 2 :]
        result["copying"] = dict(
            generated=data_copying(train_l, real_l, gen_l, seed=seed),
            reference=data_copying(train_l, real_l[b], real_l[a], seed=seed),
        )
    real_d, gen_d = (
        np.array([descriptors(c) for c in real]),
        np.array([descriptors(c) for c in generated]),
    )
    result["descriptors"] = {
        n: dict(
            w1=float(wasserstein_distance(gen_d[:, j], real_d[:, j])),
            median_generated=float(np.median(gen_d[:, j])),
            median_real=float(np.median(real_d[:, j])),
        )
        for j, n in enumerate(("solidity", "width/length", "circularity"))
    }
    return result


def _iv(iv: dict, key: str) -> str:
    v = iv.get(key)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]:.2f}–{v[1]:.2f}"
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


def markdown(label: str, r: dict) -> str:
    lines = [
        f"# Outline evaluation: {label}",
        "",
        f"{r['kind']} model `{r['model']}`, {r['generated']} draws, against {r['real']} real "
        f"outlines of `{r['reference']}`.",
        "",
        "| space | rows | density ↑ (1) | coverage ↑ | 1-NNA (0.5) |",
        "|---|---|---|---|---|",
    ]
    for space in ("landmark", "efd_pca16"):
        s = r[space]
        m, iv, ref = s["metrics"], s["interval"], s["real_vs_real"]
        lines += [
            f"| {space} | generated vs real | {m['density']:.2f} | {m['coverage']:.2f} | "
            f"{m['one_nna']:.2f} |",
            f"| {space} | 80 % subsampling interval | {_iv(iv, 'density')} | "
            f"{_iv(iv, 'coverage')} | {_iv(iv, 'one_nna')} |",
        ]
        if "interval_95" in s:
            ci = s["interval_95"]
            lines.append(
                f"| {space} | 95 % interval (Politis & Romano) | {_iv(ci, 'density')} | "
                f"{_iv(ci, 'coverage')} | {_iv(ci, 'one_nna')} |"
            )
        lines += [
            f"| {space} | real half vs real half | {_iv(ref, 'density')} | "
            f"{_iv(ref, 'coverage')} | {_iv(ref, 'one_nna')} |",
        ]
    if r.get("copying"):
        lines += [
            "",
            f"Data copying (Meehan et al. 2020) in the landmark space against `{r['train']}`:",
            "",
            "| rows | Z_U (<<0 copies) | C_T (cells) | median DCR | NN ratio < 1/3 |",
            "|---|---|---|---|---|",
        ]
        for name, c in r["copying"].items():
            lines.append(
                f"| {name} | {c['z_u']:.1f} | {c['c_t']:.1f} ({c['cells_used']}/{c['cells']}) | "
                f"{c['dcr_median_generated']:.3f} (held-out {c['dcr_median_held']:.3f}) | "
                f"{c['nn_ratio_below_third']:.1%} |"
            )
    lines += ["", "| descriptor | W1 | median generated | median real |", "|---|---|---|---|"]
    for n, d in r["descriptors"].items():
        lines.append(
            f"| {n} | {d['w1']:.4f} | {d['median_generated']:.3f} | {d['median_real']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def table() -> str:
    """One row per <label>.json of the output folder."""
    rows = []
    # labels <subset>-<row>: grouped by subset, the real row first
    paths = sorted(
        OUT.glob("*.json"), key=lambda p: (p.stem.split("-")[0], "real" not in p.stem, p.stem)
    )
    for path in paths:
        r = json.loads(path.read_text())
        lm = r.get("landmark", {}).get("metrics", {})
        ef = r["efd_pca16"]["metrics"]
        c = (r.get("copying") or {}).get("generated", {})
        rows.append(
            f"| {path.stem} | {r['kind']} | {r['generated']} | {_iv(lm, 'density')} | "
            f"{_iv(lm, 'coverage')} | {_iv(lm, 'one_nna')} | {_num(c.get('z_u'), 1)} | "
            f"{_num(c.get('c_t'), 1)} | {_pct(c.get('nn_ratio_below_third'))} | "
            f"{ef['one_nna']:.2f} | {r['descriptors']['circularity']['w1']:.3f} |"
        )
    lines = [
        "# 2D shape generators against the real outlines",
        "",
        "Set metrics in the pseudo-landmark space (Naeem et al. 2020 density / coverage, k = 5; "
        "1-NNA, 0.5 = indistinguishable; ideal density 1, coverage ~0.97), data copying against "
        "the training bank (Meehan et al. 2020: Z_U << 0 copies, >> 0 underfits, ideal ~0; C_T the "
        "same over k-means cells; share of draws with nearest/second-nearest training ratio "
        "< 1/3), 1-NNA in EFD-PCA-16 and W1 of circularity. Read every row against the `real_*` "
        "row of its subset.",
        "",
        "| label | kind | draws | density ↑ | coverage ↑ | 1-NNA (0.5) | Z_U (~0) | C_T (~0) | "
        "NN ratio < 1/3 | 1-NNA EFD-PCA-16 | W1 circularity ↓ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text)
    return text


def _num(v, digits: int) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "–"


def _pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "–"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="model folder or train:<bank>")
    parser.add_argument("--reference", type=Path, help="holdout bank .npz or canonical/ folder")
    parser.add_argument("--train", type=Path, default=None, help="bank the model was fitted on")
    parser.add_argument("--count", type=int, default=None, help="draws (default: reference size)")
    parser.add_argument("--subset", default=None, help="subset to condition a pooled model on")
    parser.add_argument(
        "--steps", type=int, default=None, help="ODE steps of a sampler (default 50)"
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", action="store_true", help="only rebuild summary.md")
    args = parser.parse_args()
    if args.table:
        print(table())
        return
    if not args.model or not args.reference:
        parser.error("--model and --reference are required")
    label = args.label or Path(args.model.split(":")[-1]).stem
    result = evaluate(
        args.model, args.reference, args.count, args.seed, args.train, args.subset, args.steps
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.json").write_text(json.dumps(result, indent=2))
    text = markdown(label, result)
    (OUT / f"{label}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
