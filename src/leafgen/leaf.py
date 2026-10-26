"""One 3D leaflet from the four stages: outline, veins, relief, bend.

    outline  a shape model draws a contour in the canonical frame (base (0, 0), tip (0, 1))
    veins    VeinFlow draws the veins inside the outline mask (256 px), the drawing is repaired
             (loops cut, detached pieces joined) and turned into a drainage tree
    relief   the tree is baked into height maps of both sides (pillow style) and an albedo
    bend     a flat triangulation of the outline gets its leaf coordinates (s, t) and is lifted
             by a height field drawn from the Pheno4D prior; the relief rides along the normal

Lengths: the canonical frame has base-tip length 1; the mesh is returned in millimetres, the leaf
``length_mm`` long (the relief maps are baked for a 60 mm leaf and kept in millimetres).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize

from leafgen.bend.bend_prior import BendPriorDeformer, BendPriorProvider
from leafgen.contracts.geometry import LeafShape2D
from leafgen.mesh.triangle import TriangleMeshBuilder
from leafgen.semantics.mean_value import MeanValueSemanticMapper
from leafgen.shape.models import MODELS, neural
from leafgen.veins import metrics, raster
from leafgen.veins import model as veinflow
from leafgen.veins.connect import bridge_loopcut
from leafgen.veins.raster import BOX_X, BOX_Y
from leafgen.veins.relief import ReliefMaps, ReliefProfile, albedo, bake
from leafgen.veins.tree import VeinTree, drainage_tree, nearest_inside

TAPER_MM = 5.0  # the two sides meet at the outline, full thickness this far inside


def stream_seed(seed: int, stream: int) -> int:
    """Independent random streams (outline, veins, bend) from one public seed."""
    return int(np.random.SeedSequence([seed, stream]).generate_state(1, dtype=np.uint32)[0])


def raster_tree(veins: np.ndarray, mask: np.ndarray, up: int = 2) -> VeinTree:
    """VeinFlow drawing (float in [0, 1], 256 px, shape frame) -> tree in the shape frame.

    The drawing is upsampled x2 (nearest), its veins skeletonised and joined into a drainage tree
    from the base (``tree.drainage_tree``, as for the photos); the edges that bridge gaps across
    the lamina are dropped, and the order VeinFlow drew caps the flow, so a vein drawn thin stays
    thin."""
    v = veins.astype(np.float32) / (255.0 if veins.dtype == np.uint8 else 1.0)
    size = v.shape[0] * up
    v = cv2.resize(v, (size, size), interpolation=cv2.INTER_NEAREST)
    m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    v = v * m
    sinks = skeletonize((v > metrics.VEIN) & m)

    def rc(x, y):
        col = (x - BOX_X[0]) / (BOX_X[1] - BOX_X[0]) * size
        row = (y - BOX_Y[0]) / (BOX_Y[1] - BOX_Y[0]) * size
        return nearest_inside(m, np.array([col, row]))

    midrib = cv2.GaussianBlur((v > metrics.MIDRIB).astype(np.float32), (0, 0), up)
    tree = drainage_tree(
        np.clip(v / 0.45, 0, 1),
        m,
        rc(0, 0),
        sinks,
        tip_rc=rc(0, 1),
        midrib_map=v + midrib,
        taper_px=1.0,
    )
    cp = np.round(tree.child).astype(int)
    keep = ((v > metrics.VEIN) | (midrib > 0.5))[cp[:, 1], cp[:, 0]]  # no bridges across lamina
    val = v[cp[:, 1], cp[:, 0]]
    top = tree.flow.max()
    cap = np.select(
        [val > metrics.MIDRIB, val > metrics.SECONDARY, val >= metrics.FINE_TOP],
        [top, 0.2 * top, 0.03 * top],
        0.006 * top,
    )
    span = np.array([BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0]])
    tree = VeinTree(tree.child[keep], tree.parent[keep], np.minimum(tree.flow, cap)[keep])
    return tree.mapped(lambda p: np.asarray(p, float) * span / size + [BOX_X[0], BOX_Y[0]])


def sample_maps(img: np.ndarray, box, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear value of a map (rows from y1 down to y0, columns from x0 to x1) at points."""
    h, w = img.shape
    c = (x - box[0]) / (box[1] - box[0]) * w - 0.5
    r = (box[3] - y) / (box[3] - box[2]) * h - 0.5
    return map_coordinates(img.astype(np.float64), [r, c], order=1, mode="nearest")


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    f = faces.astype(np.int64)
    n = np.cross(vertices[f[:, 1]] - vertices[f[:, 0]], vertices[f[:, 2]] - vertices[f[:, 0]])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@dataclass
class Leaf:
    seed: int
    contour: np.ndarray  # canonical frame
    drawing: np.ndarray  # repaired VeinFlow drawing, 256 px
    tree: VeinTree
    maps: ReliefMaps
    flat_xy: np.ndarray  # mesh vertices, canonical frame
    faces: np.ndarray  # counter-clockwise seen from the upper side
    boundary: np.ndarray  # indices of the vertices on the outline
    bent: np.ndarray  # mm, before the relief
    normals: np.ndarray
    upper: np.ndarray  # mm, upper side with its relief
    lower: np.ndarray  # mm, lower side with its ribs (same faces, reversed)
    uv: np.ndarray  # texture coordinates of the relief maps
    thickness: np.ndarray  # 0 at the outline, 1 inside, per vertex


class LeafGenerator:
    """Loads the three models once; ``leaf(seed)`` gives one leaflet."""

    def __init__(
        self,
        weights: Path,
        *,
        shape_model: str = "rae64_flow",
        subset: str = "field",
        bend_temperature: float = 0.9,
        mesh_area: float = 2e-5,
        relief_resolution: int = 2048,
        length_mm: float = 60.0,
        veins: bool = True,
    ) -> None:
        weights = Path(weights)
        folder = weights / shape_model
        name = json.loads((folder / "metadata.json").read_text())["model"]
        self.shape = MODELS[name].Generator(folder, subset=subset)
        self.subset = subset
        self.veins = veins
        if veins:
            import torch

            meta = json.loads((weights / "veinflow/metadata.json").read_text())
            net, _ = veinflow.build(meta["size"], tuple(meta["widths"]), meta.get("layers", 1))
            net.load_state_dict(torch.load(weights / "veinflow/model.pt", map_location="cpu"))
            self.device = neural.device()
            self.net = net.to(self.device).eval()
            self.torch = torch
            self.size = meta["size"]
        provider = BendPriorProvider.load(weights / "bend_prior.npz", temperature=bend_temperature)
        self.provider = provider
        self.deformer = BendPriorDeformer(provider=provider)
        self.mesher = TriangleMeshBuilder(options=f"pq20a{mesh_area:.8f}".rstrip("0"))
        self.mapper = MeanValueSemanticMapper()
        self.profile = ReliefProfile(style="pillow")
        self.relief_resolution = relief_resolution
        self.length_mm = length_mm

    def draw_veins(self, contour: np.ndarray, seed: int) -> tuple[np.ndarray, VeinTree]:
        mask = raster.outline_mask(contour, self.size)
        subset = np.array([veinflow.SUBSETS.index(self.subset)])
        drawing = veinflow.draw(self.net, self.torch, mask[None], subset, seed, self.device)[0]
        drawing = bridge_loopcut(drawing, mask)
        return drawing, raster_tree(drawing, mask).pruned(3)

    def leaf(self, seed: int) -> Leaf:
        contour = self.shape.sample(1, stream_seed(seed, 0))[0]
        if self.veins:
            drawing, tree = self.draw_veins(contour, stream_seed(seed, 1))
            maps = bake(tree, contour, profile=self.profile, resolution=self.relief_resolution)
        else:  # smooth lamina: the maps of a one-segment tree, heights and veins cleared
            drawing = np.zeros((256, 256), np.float32)
            tree = VeinTree(np.array([[0.0, 0.1]]), np.array([[0.0, 0.0]]), np.ones(1))
            maps = bake(tree, contour, profile=self.profile, resolution=self.relief_resolution)
            zero = np.zeros_like(maps.top_mm)
            maps = replace(maps, top_mm=zero, bottom_mm=zero, vein=zero)
        shape = LeafShape2D(contour, np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        mesh = self.mesher.build(shape)
        semantic = self.mapper.compute(shape, mesh)
        sample = self.provider.sample(seed=stream_seed(seed, 2), role="terminal")
        lifted, *_ = self.deformer.apply(semantic, sample)
        bent = lifted * self.length_mm
        faces = np.asarray(mesh.faces, dtype=np.int64)
        normals = vertex_normals(bent, faces)
        x, y = mesh.vertices_xy.T
        boundary = np.unique(mesh.boundary_edges)
        # distance to the outline, in mm, from the relief mask
        dist = cv2.distanceTransform(maps.mask.astype(np.uint8), cv2.DIST_L2, 5)
        dist *= (maps.box[3] - maps.box[2]) * self.profile.leaf_mm / maps.mask.shape[0]
        u = np.clip(sample_maps(dist, maps.box, x, y) / TAPER_MM, 0, 1)
        u[boundary] = 0
        thickness = np.sqrt(1 - (1 - u) ** 2)  # rounded lens profile near the outline
        top = sample_maps(maps.top_mm, maps.box, x, y) * u
        bottom = sample_maps(maps.bottom_mm, maps.box, x, y) * u
        half = maps.lamina_mm / 2 * thickness
        upper = bent + normals * (half + top)[:, None]
        lower = bent - normals * (half + bottom)[:, None]
        uv = np.column_stack(
            [
                (x - maps.box[0]) / (maps.box[1] - maps.box[0]),
                (y - maps.box[2]) / (maps.box[3] - maps.box[2]),
            ]
        )
        return Leaf(
            seed,
            contour,
            drawing,
            tree,
            maps,
            mesh.vertices_xy,
            faces,
            boundary,
            bent,
            normals,
            upper,
            lower,
            uv,
            thickness,
        )


def write_obj(leaf: Leaf, folder: Path, name: str) -> Path:
    """``name.obj`` (two sides sharing the outline, one closed surface), ``name.mtl`` and the
    albedo of both sides as PNG textures."""
    folder.mkdir(parents=True, exist_ok=True)
    for side in ("top", "bottom"):
        rgb = (np.clip(albedo(leaf.maps, side), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{name}_{side}.png"), rgb[..., ::-1])
    (folder / f"{name}.mtl").write_text(
        "".join(
            f"newmtl {side}\nKd 1 1 1\nmap_Kd {name}_{side}.png\n\n" for side in ("top", "bottom")
        )
    )
    n = len(leaf.upper)
    # the lower side reuses the outline vertices of the upper one, so the surface is closed
    inner = np.setdiff1d(np.arange(n), leaf.boundary)
    lower_index = np.arange(n)
    lower_index[inner] = n + np.arange(len(inner))
    lines = [f"mtllib {name}.mtl", f"o {name}"]
    lines += [f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in leaf.upper]
    lines += [f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in leaf.lower[inner]]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in leaf.uv]
    lines.append("usemtl top")
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in leaf.faces + 1]
    lines.append("usemtl bottom")
    for a, b, c in leaf.faces:
        la, lb, lc = lower_index[[a, c, b]] + 1
        lines.append(f"f {la}/{a + 1} {lb}/{c + 1} {lc}/{b + 1}")
    path = folder / f"{name}.obj"
    path.write_text("\n".join(lines) + "\n")
    return path
