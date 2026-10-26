"""UV maps of the vein relief, computed on the flat leaf and applied after the 3D bend.

Two styles:

- ``pillow`` (default), tuned on close-up photos of tomato leaves. On top, each areole between
  veins bulges like a quilt pillow (higher for larger areoles) and the veins are the narrow
  creases between pillows. A faint reticulate network of minor veins (warped Voronoi cells,
  ~2.2 mm) adds the small quilting. Underneath, the veins are narrow rounded ribs standing out,
  paler than the lamina, and the areoles bulge outwards a little too. Lamina 0.40 mm.
- ``groove``, the first version: a wide Gaussian valley on top and a ridge underneath,
  h(r) = depth * exp(-(r / width)^2), lamina 0.25 mm.

Width and depth grow with the flow, so the midrib is the deepest. Heights are in mm along the
surface normal, outwards on each side, so the bend moves them with the lamina: displace the two
sides along the bent normal, or use the maps as bump/normal maps. The sizes are plausible guesses
checked by eye against photos, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from leafgen.veins.tree import VeinTree


@dataclass(frozen=True)
class ReliefProfile:
    style: str = "pillow"  # or "groove"
    leaf_mm: float = 60.0  # base-tip length
    lamina_bumps_mm: float = 0.012  # fine random texture
    classes: int = 8  # veins are drawn in flow classes
    # groove style: a + b * q^k, q = log flow normalised to [0, 1]
    width_mm: tuple[float, float] = (0.12, 0.75)  # half-width, k = 2.2
    groove_mm: tuple[float, float] = (0.02, 0.22)  # top depth, k = 2.0
    ridge_mm: tuple[float, float] = (0.03, 0.70)  # bottom height, k = 1.8
    # pillow style
    lamina_mm: float = 0.40  # thickness between the two sides
    crease_width_mm: tuple[float, float] = (0.05, 0.30)  # k = 1.5
    crease_mm: tuple[float, float] = (0.04, 0.30)  # k = 1.5
    rib_width_mm: tuple[float, float] = (0.10, 0.55)  # k = 1.8
    rib_mm: tuple[float, float] = (0.04, 0.85)  # k = 1.6
    pillow_ratio: float = 0.10  # areole bulge height / areole radius
    pillow_max_mm: float = 0.30
    minor_cell_mm: float = 2.2  # spacing of the minor-vein network
    minor_crease_mm: float = 0.012
    minor_rib_mm: float = 0.018


@dataclass(frozen=True)
class ReliefMaps:
    box: tuple[float, float, float, float]  # canonical x0, x1, y0, y1 covered by the maps
    mask: np.ndarray
    top_mm: np.ndarray  # signed height, negative = crease
    bottom_mm: np.ndarray  # positive = rib (outwards, away from the top)
    vein: np.ndarray  # 0..1, for albedo
    lamina_mm: float = 0.40  # separation of the two sides the heights are added to
    style: str = "pillow"


def bake(
    tree: VeinTree,
    outline: np.ndarray,
    *,
    profile: ReliefProfile | None = None,
    box: tuple[float, float, float, float] = (-0.5, 0.5, -0.08, 1.08),
    resolution: int = 2048,
    seed: int = 0,
) -> ReliefMaps:
    """Rows go from y1 (top of the image) down to y0, columns from x0 to x1."""
    profile = profile or ReliefProfile()
    if profile.style not in ("pillow", "groove"):
        raise ValueError(f"unknown relief style {profile.style!r}")
    h = resolution
    w = int(round(resolution * (box[1] - box[0]) / (box[3] - box[2])))

    def to_px(p):
        p = np.asarray(p, float)
        x = (p[:, 0] - box[0]) / (box[1] - box[0]) * w
        return np.stack([x, (box[3] - p[:, 1]) / (box[3] - box[2]) * h], 1)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(to_px(outline) * 4).astype(np.int32)], 1, shift=2)
    mask = mask > 0
    px_per_mm = h / ((box[3] - box[2]) * profile.leaf_mm)
    q = np.log(np.maximum(tree.flow, 1)) / np.log(max(tree.flow.max(), 2))
    a, b = to_px(tree.child), to_px(tree.parent)
    rng = np.random.default_rng(seed)
    bumps = ndi.gaussian_filter(rng.standard_normal((h, w)), 0.35 * px_per_mm)
    bumps *= profile.lamina_bumps_mm / bumps.std()
    edge = np.clip(ndi.distance_transform_edt(mask) / (0.8 * px_per_mm), 0, 1)  # flat margin
    if profile.style == "groove":
        top, bottom, vein = _groove(a, b, q, (h, w), px_per_mm, profile)
        top, bottom, lamina = -top + bumps, bottom + 0.5 * bumps, 0.25
    else:
        top, bottom, vein = _pillow(a, b, q, mask, px_per_mm, profile, rng)
        top, bottom, lamina = top + 0.5 * bumps, bottom + 0.5 * bumps, profile.lamina_mm
    return ReliefMaps(
        box=box,
        mask=mask,
        top_mm=(top * mask * edge).astype(np.float32),
        bottom_mm=(bottom * mask * edge).astype(np.float32),
        vein=(vein * mask).astype(np.float32),
        lamina_mm=lamina,
        style=profile.style,
    )


def _distance_mm(a, b, shape, px_per_mm):
    """Distance in mm from every pixel to the polyline segments a[i] -> b[i] (pixel xy)."""
    lines = np.zeros(shape, np.uint8)
    for p, r in zip(np.round(a * 4).astype(int), np.round(b * 4).astype(int), strict=True):
        cv2.line(lines, tuple(p), tuple(r), 1, 1, cv2.LINE_8, shift=2)
    return ndi.distance_transform_edt(lines == 0) / px_per_mm


def _groove(a, b, q, shape, px_per_mm, profile):
    width = profile.width_mm[0] + profile.width_mm[1] * q**2.2
    groove = profile.groove_mm[0] + profile.groove_mm[1] * q**2.0
    ridge = profile.ridge_mm[0] + profile.ridge_mm[1] * q**1.8
    cls = np.minimum((q * profile.classes).astype(int), profile.classes - 1)
    top, bottom, vein = (np.zeros(shape, np.float32) for _ in range(3))
    for c in range(profile.classes):
        sel = cls == c
        if not sel.any():
            continue
        r = _distance_mm(a[sel], b[sel], shape, px_per_mm)
        shape_c = np.exp(-((r / width[sel].mean()) ** 2))
        top = np.maximum(top, groove[sel].mean() * shape_c)
        bottom = np.maximum(bottom, ridge[sel].mean() * shape_c)
        vein = np.maximum(vein, np.sqrt(q[sel].mean()) * shape_c)
    return top, bottom, vein


def _pillow(a, b, q, mask, px_per_mm, profile, rng):
    shape = mask.shape
    keep = _drop_spurs(a, b, 1.2 * px_per_mm)
    a, b = a[keep], b[keep]
    q = _smooth_along(a, b, q[keep])
    a, b = _smooth_segments(a, b)
    cls = np.minimum((q * profile.classes).astype(int), profile.classes - 1)
    crease, rib, vein = (np.zeros(shape, np.float32) for _ in range(3))
    near = np.full(shape, np.inf, np.float32)
    for c in range(profile.classes):
        own = cls == c
        if not own.any():
            continue
        qc = q[own].mean()
        # draw the larger veins too, so there are no gaps where a vein changes class
        sel = cls >= c
        # blur: the 1-px raster staircase would ripple the rib slopes
        r = ndi.gaussian_filter(_distance_mm(a[sel], b[sel], shape, px_per_mm), 2.0)
        near = np.minimum(near, r)
        wg = profile.crease_width_mm[0] + profile.crease_width_mm[1] * qc**1.5
        depth = profile.crease_mm[0] + profile.crease_mm[1] * qc**1.5
        wr = profile.rib_width_mm[0] + profile.rib_width_mm[1] * qc**1.8
        height = profile.rib_mm[0] + profile.rib_mm[1] * qc**1.6
        crease = np.maximum(crease, depth * np.exp(-((r / wg) ** 2)))
        rib = np.maximum(rib, height * np.exp(-((r / wr) ** 2.5)))
        vein = np.maximum(vein, np.sqrt(qc) * np.exp(-((r / wg) ** 2)))
    # minor veins: edges of a jittered Voronoi tessellation, warped so they wander
    h, w = shape
    step = profile.minor_cell_mm * px_per_mm
    gy, gx = np.mgrid[0:h:step, 0:w:step]
    pts = np.stack([gy.ravel(), gx.ravel()], 1) + rng.uniform(-0.45, 0.45, (gy.size, 2)) * step
    pts = np.clip(pts, 0, [h - 1, w - 1]).astype(int)
    seeds = np.ones(shape, np.uint8)
    seeds[pts[:, 0], pts[:, 1]] = 0
    _, cell = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    warp = [ndi.gaussian_filter(rng.standard_normal(shape), 0.6 * step) for _ in range(2)]
    warp = [(0.35 * step * g / g.std()).astype(np.float32) for g in warp]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cell = cv2.remap(cell.astype(np.float32), xx + warp[1], yy + warp[0], cv2.INTER_NEAREST)
    edges = np.zeros(shape, bool)
    edges[:, 1:] |= cell[:, 1:] != cell[:, :-1]
    edges[1:, :] |= cell[1:, :] != cell[:-1, :]
    r_minor = ndi.distance_transform_edt(~edges) / px_per_mm
    minor = np.exp(-((r_minor / 0.07) ** 2))
    crease = np.maximum(crease, profile.minor_crease_mm * minor)
    rib = np.maximum(rib, profile.minor_rib_mm * np.exp(-((r_minor / 0.06) ** 2.5)))
    vein = np.maximum(vein, 0.12 * minor)
    # pillows: one per areole bounded by the veins, height grows with the areole radius
    areole, n = ndi.label((near > profile.crease_width_mm[0]) & mask)
    radius = ndi.maximum(near, areole, index=np.arange(1, n + 1)) if n else np.array([])
    big = np.r_[1.0, radius][areole]
    u = np.clip(near / np.maximum(big, 1e-3), 0, 1)
    height = np.minimum(profile.pillow_ratio * big, profile.pillow_max_mm)
    dome = ndi.gaussian_filter(height * np.sqrt(1 - (1 - u) ** 2) * (areole > 0), 0.06 * px_per_mm)
    u = np.clip(r_minor / (0.5 * profile.minor_cell_mm), 0, 1)
    dome = dome + 0.02 * np.sqrt(1 - (1 - u) ** 2)  # small pillows on the minor cells
    # creases are only a few pixels wide: antialias them
    crease, rib = ndi.gaussian_filter(crease, 1.2), ndi.gaussian_filter(rib, 1.2)
    return dome - crease, rib + 0.3 * dome, vein


def _graph(a, b):
    """Node index of both ends of every segment, merging coincident endpoints."""
    pts, inv = np.unique(np.round(np.vstack([a, b]), 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    return pts, inv[: len(a)], inv[len(a) :]


def _drop_spurs(a, b, min_len):
    """Mask of segments to keep: drops terminal branches shorter than min_len (pixels).

    The drainage tree has short side spurs along the big veins (every bright pixel drains into
    them); drawn as relief they look like stripes across the vein."""
    pts, i, j = _graph(a, b)
    seg_len = np.hypot(*(a - b).T)
    deg = np.bincount(np.r_[i, j], minlength=len(pts))
    nbrs = [[] for _ in pts]
    for s, (u, v) in enumerate(zip(i, j, strict=True)):
        nbrs[u].append((v, s))
        nbrs[v].append((u, s))
    keep = np.ones(len(a), bool)
    for tip in np.flatnonzero(deg == 1):
        path, length, prev, cur = [], 0.0, -1, tip
        while deg[cur] <= 2 and length < min_len:
            nxt = [(v, s) for v, s in nbrs[cur] if v != prev]
            if not nxt:
                break
            v, s = nxt[0]
            path.append(s)
            length += seg_len[s]
            prev, cur = cur, v
        if deg[cur] >= 3 and length < min_len:
            keep[path] = False
    return keep


def _smooth_along(a, b, q, iters=25, same=0.15):
    """Average q between segments that meet with a similar q (the same vein), so the flow class
    does not flicker along a vein; junctions with much thinner branches are left alone."""
    pts, i, j = _graph(a, b)
    q = q.astype(float).copy()
    for _ in range(iters):
        node_max = np.zeros(len(pts))
        np.maximum.at(node_max, i, q)
        np.maximum.at(node_max, j, q)
        acc, cnt = np.zeros(len(pts)), np.zeros(len(pts))
        for n in (i, j):
            ok = q > node_max[n] - same
            np.add.at(acc, n[ok], q[ok])
            np.add.at(cnt, n[ok], 1)
        node_q = acc / np.maximum(cnt, 1)
        at_i = np.where(q > node_max[i] - same, node_q[i], q)
        at_j = np.where(q > node_max[j] - same, node_q[j], q)
        q = (q + at_i + at_j) / 3
    return q


def _smooth_segments(a, b, iters=6, lam=0.5):
    """Laplacian smoothing of the tree drawn in pixels: removes the pixel-graph staircase.
    Tips, branch points and the base stay fixed."""
    pts, i, j = _graph(a, b)
    deg = np.bincount(np.r_[i, j], minlength=len(pts))
    free = deg == 2
    p = pts.astype(float)
    for _ in range(iters):
        acc = np.zeros_like(p)
        np.add.at(acc, i, p[j])
        np.add.at(acc, j, p[i])
        avg = acc / np.maximum(deg, 1)[:, None]
        p[free] += lam * (avg[free] - p[free])
    return p[i], p[j]


def albedo(maps: ReliefMaps, side: str, *, seed: int = 0) -> np.ndarray:
    """Linear RGB in [0, 1]: darker lamina with lighter veins on top, paler underneath."""
    if side == "top":
        lamina, veins, mix = np.array([0.045, 0.11, 0.025]), np.array([0.10, 0.18, 0.05]), 0.45
    elif maps.style == "pillow":  # photos: underside veins clearly paler, whitish green
        lamina, veins, mix = np.array([0.13, 0.21, 0.09]), np.array([0.42, 0.50, 0.30]), 0.95
    else:
        lamina, veins, mix = np.array([0.12, 0.20, 0.08]), np.array([0.30, 0.38, 0.20]), 0.7
    rng = np.random.default_rng(seed + 1)
    px_per_unit = maps.mask.shape[0] / (maps.box[3] - maps.box[2])
    tone = ndi.gaussian_filter(rng.standard_normal(maps.mask.shape), 0.04 * px_per_unit)
    tone = 1 + 0.08 * tone / tone.std()
    v = (maps.vein * mix)[..., None]
    return np.clip(lamina * tone[..., None] * (1 - v) + veins * v, 0, 1)


def save_png(maps: ReliefMaps, prefix: str) -> None:
    """<prefix>_top.png / _bottom.png: sRGB albedo + alpha; <prefix>_htop.png / _hbottom.png:
    16-bit heights, 32768 = 0, full range +-1 mm."""
    alpha = (maps.mask * 255).astype(np.uint8)
    for side, height in (("top", maps.top_mm), ("bottom", maps.bottom_mm)):
        srgb = (np.clip(albedo(maps, side), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        cv2.imwrite(f"{prefix}_{side}.png", np.dstack([srgb[..., ::-1], alpha]))
        cv2.imwrite(
            f"{prefix}_h{side}.png", np.clip((height / 2 + 0.5) * 65535, 0, 65535).astype(np.uint16)
        )
