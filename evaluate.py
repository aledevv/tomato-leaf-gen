"""Reproduce the evaluation tables.

    uv run evaluate.py               # outlines and veins, the five subsets
    uv run evaluate.py --quick       # chamber only, outlines without mask diffusion, vein statistics
    uv run evaluate.py --only veins  # or shape

Outlines (fresh 75/25 split by photo, ``final75``): every row draws as many outlines as the
held-out bank holds (seed 0) and is measured in the pseudo-landmark space (density / coverage,
Naeem et al. 2020; 1-NNA, Lopez-Paz & Oquab 2017; data copying Z_U, Meehan et al. 2020). Rows: real
training leaflets (the ceiling), RAE-64 + flow, mask diffusion. The draws are made again here:
the random stream depends on the device (the paper drew on a GPU), so the model rows move a little:
on a CPU, RAE-64 + flow on chamber gives 1-NNA 0.575 against the published 0.591. The real row is
exact. Published values and their 95 % subsampling interval: ``expected/shape.json``.

Veins (split75, 500 held-out leaflets per subset, seed 0): the VeinFlow drawings of the paper
(``data/drawings``, drawn once on a GPU) and the same drawings repaired here
(``veins.connect.bridge_loopcut``), against the real held-out rasters; ``ideal`` = as many real
training leaflets. Spaces: the 16 statistics of ``veins.metrics`` (with and without the vein orders),
the botanical characters (Ellis et al. 2009, vein length per area) and DINOv2 ViT-L/14
(Stein et al. 2023). These rows reproduce exactly. ``--redraw`` draws new VeinFlow veins instead
(a GPU is advised).

Writes ``results/<part>.json`` and prints each row next to its published value.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from leafgen.paths import DATA, ROOT, WEIGHTS

SUBSETS = ("chamber", "field", "BILs", "tomato_entire", "tomato_asymmetry")
OUT = ROOT / "results"


def shape(subsets, rows) -> list[dict]:
    from leafgen.shape.evaluate import evaluate

    expected = json.loads((ROOT / "expected/shape.json").read_text())
    out = []
    for s in subsets:
        bank = DATA / f"sam3_{s}_models"
        for row in rows:
            model = f"train:{bank / 'final75.npz'}" if row == "real" else str(WEIGHTS / row)
            t0 = time.time()
            r = evaluate(model, bank / "final75_holdout.npz", None, 0, bank / "final75.npz", s)
            m, z = r["landmark"]["metrics"], r["copying"]["generated"]["z_u"]
            e = expected[s]["real" if row == "real" else row]
            lo, hi = e["interval_95"]["one_nna"]
            out.append(
                dict(subset=s, row=row, **m, z_u=z, interval_95=r["landmark"]["interval_95"])
            )
            print(
                f"shape {s:16} {row:14} density {m['density']:.2f} ({e['density']:.2f})  "
                f"coverage {m['coverage']:.2f} ({e['coverage']:.2f})  1-NNA {m['one_nna']:.3f} "
                f"(paper {e['one_nna']:.3f}, 95 % {lo:.2f}-{hi:.2f})  Z_U {z:.1f} ({e['z_u']:.1f})  "
                f"[{time.time() - t0:.0f} s]",
                flush=True,
            )
    return out


def veins(subsets, spaces, redraw: bool) -> list[dict]:
    from leafgen.veins import known
    from leafgen.veins.connect import bridge_loopcut

    expected = {
        (e["subset"], e["label"], e["space"], e["binary"]): e
        for e in json.loads((ROOT / "expected/veins.json").read_text())
    }
    out = []
    for s in subsets:
        folder = DATA / f"sam3_{s}_veins"
        keys = ("instance_id", "veins", "masks")
        with np.load(folder / "raster256_split75.npz") as f:
            train = {k: f[k] for k in keys}
        with np.load(folder / "raster256_split75_holdout.npz") as f:
            held = {k: f[k] for k in keys}
        rng = np.random.default_rng(0)  # as in the paper: first the held-out pick, then the ideal
        pick = np.sort(rng.permutation(len(held["instance_id"]))[:500])
        some = np.sort(rng.permutation(len(train["instance_id"]))[: len(pick)])
        ids, masks = held["instance_id"][pick], held["masks"][pick]
        if redraw:
            drawn = draw_veinflow(s, masks)
        else:
            g = np.load(DATA / "drawings" / f"veinflow_{s}.npz")
            row = {k: i for i, k in enumerate(g["instance_id"])}
            drawn = g["veins"][[row[k] for k in ids]]
        repaired = np.array(
            [(bridge_loopcut(v, m) * 255).round().astype(np.uint8) for v, m in zip(drawn, masks, strict=True)]
        )
        sets = {
            "ideal": (train["veins"][some], train["masks"][some]),
            "VeinFlow": (drawn, masks),
            "VeinFlow+repair": (repaired, masks),
        }
        for space in spaces:
            for binary in (False, True):
                real = known.table(space, held["veins"][pick], masks, True, binary)
                for label, (images, m) in sets.items():
                    gen = known.table(space, images, m, True, binary)
                    r = known.compare(space, gen, real)
                    e = expected.get((s, label, space, binary), {})
                    out.append(dict(subset=s, label=label, space=space, binary=binary, **r))
                    tag = f"{space}{' order-free' if binary else ''}"
                    print(
                        f"veins {s:16} {tag:18} {label:16} 1-NNA {r['one_nna']:.3f} "
                        f"(paper {e.get('one_nna', float('nan')):.3f})",
                        flush=True,
                    )
    return out


def draw_veinflow(subset: str, masks: np.ndarray) -> np.ndarray:
    import torch

    from leafgen.shape.models import neural
    from leafgen.veins import model as vf

    meta = json.loads((WEIGHTS / "veinflow/metadata.json").read_text())
    net, _ = vf.build(meta["size"], tuple(meta["widths"]), meta.get("layers", 1))
    net.load_state_dict(torch.load(WEIGHTS / "veinflow/model.pt", map_location="cpu"))
    device = neural.device()
    net.to(device).eval()
    subsets = np.full(len(masks), vf.SUBSETS.index(subset))
    drawn = vf.draw(net, torch, masks, subsets, 0, device)
    return (drawn * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="chamber only, the fast rows")
    parser.add_argument("--only", choices=("shape", "veins"), default=None)
    parser.add_argument("--redraw", action="store_true", help="draw new VeinFlow veins")
    args = parser.parse_args()
    subsets = ("chamber",) if args.quick else SUBSETS
    OUT.mkdir(exist_ok=True)
    if args.only in (None, "shape"):
        rows = ("real", "rae64_flow") if args.quick else ("real", "rae64_flow", "mask_diffusion")
        result = shape(subsets, rows)
        (OUT / ("shape_quick.json" if args.quick else "shape.json")).write_text(
            json.dumps(result, indent=1)
        )
    if args.only in (None, "veins"):
        spaces = ("stats",) if args.quick else ("stats", "botany", "dinov2")
        result = veins(subsets, spaces, args.redraw)
        (OUT / ("veins_quick.json" if args.quick else "veins.json")).write_text(
            json.dumps(result, indent=1)
        )


if __name__ == "__main__":
    main()
