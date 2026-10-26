"""Per-order vein rasters for ``model --orders``: the leaflets of ``raster<size>_<bank>.npz``, in its
order and clipped to its masks, with midrib, secondary, tertiary and finer veins on four channels
(``raster.order_layers``), as an uncompressed uint8 .npy (n, 4, size, size) read with memory
mapping (~260 KB per leaflet at 256 px).

    python -m leafgen.veins.orders --subset field --bank split75 --size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leafgen.paths import DATA
from leafgen.veins import raster
from leafgen.veins.tree import VeinTree

DERIVED = DATA


def path(subset: str, bank: str, size: int) -> Path:
    return DERIVED / f"sam3_{subset}_veins" / f"raster{size}_{bank}_orders.npy"


def write(subset: str, bank: str, size: int) -> Path:
    folder = DERIVED / f"sam3_{subset}_veins"
    r = np.load(folder / f"raster{size}_{bank}.npz")
    out = path(subset, bank, size)
    tmp = out.with_suffix(".tmp.npy")
    layers = np.lib.format.open_memmap(tmp, "w+", np.uint8, (len(r["instance_id"]), 4, size, size))
    for i, (leaflet, mask) in enumerate(zip(r["instance_id"], r["masks"], strict=True)):
        tree = VeinTree.load(folder / f"{leaflet}.npz").mapped(raster.to_shape_frame)
        layers[i] = (raster.order_layers(tree, size, fine=True) * 255).round().astype(
            np.uint8
        ) * mask
    layers.flush()
    del layers
    tmp.rename(out)
    print(f"{len(r['instance_id'])} leaflets -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", required=True)
    parser.add_argument("--bank", default="split75")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    write(args.subset, args.bank, args.size)


if __name__ == "__main__":
    main()
