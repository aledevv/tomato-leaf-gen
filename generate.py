"""Generate 3D tomato leaflets.

    uv run generate.py

Writes, for every seed, ``samples/leaf_<seed>.obj`` (closed surface: upper side with its relief,
underside with its ribs; millimetres, base at the origin, tip along +y) with its ``.mtl`` and the
albedo of both sides, the outline and veins as a PNG, and ``samples/viewer.html`` with all of them.
Change the settings below; the defaults are the models of the paper.
"""

from pathlib import Path

import cv2
import numpy as np

from leafgen.leaf import LeafGenerator, write_obj
from leafgen.paths import WEIGHTS
from leafgen.viewer import write_viewer

SETTINGS = dict(
    seeds=range(8),  # one leaflet per seed; the same seed gives the same leaflet
    shape_model="rae64_flow",  # outline model: "rae64_flow" or "mask_diffusion"
    subset="chamber",  # chamber, field, BILs, tomato_entire, tomato_asymmetry (Chitwood 2014)
    veins=True,  # VeinFlow veins and their relief; False gives a smooth lamina
    bend_temperature=0.9,  # spread of the bend prior (1 = as fitted; lower stays nearer the mean bend)
    length_mm=60.0,  # base-tip length
    mesh_area=2e-5,  # largest triangle, in units of the base-tip length squared
    output=Path("samples"),
)


def preview(leaf, path: Path, size: int = 512) -> None:
    """Outline and repaired VeinFlow drawing, as drawn (256 px, shape frame)."""
    v = cv2.resize(leaf.drawing, (size, size), interpolation=cv2.INTER_NEAREST)
    img = np.full((size, size, 3), 245, np.uint8)
    img[v > 0.05] = (40, 90, 40)
    span = size / 1.2  # the 256 px frame covers x in [-0.6, 0.6], y in [-0.1, 1.1]
    pts = np.column_stack([(leaf.contour[:, 0] + 0.6) * span, (leaf.contour[:, 1] + 0.1) * span])
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img[::-1])


def main() -> None:
    s = SETTINGS
    generator = LeafGenerator(
        WEIGHTS,
        shape_model=s["shape_model"],
        subset=s["subset"],
        veins=s["veins"],
        bend_temperature=s["bend_temperature"],
        length_mm=s["length_mm"],
        mesh_area=s["mesh_area"],
    )
    leaves = []
    for seed in s["seeds"]:
        leaf = generator.leaf(seed)
        path = write_obj(leaf, s["output"], f"leaf_{seed:03d}")
        preview(leaf, s["output"] / f"leaf_{seed:03d}_veins.png")
        leaves.append(leaf)
        print(f"{path}: {len(leaf.upper)} vertices, {len(leaf.tree.flow)} vein segments")
    print(write_viewer(leaves, s["output"] / "viewer.html", s["length_mm"]))


if __name__ == "__main__":
    main()
