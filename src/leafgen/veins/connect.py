"""Connectivity of vein drawings, and ways to make VeinFlow's drawings connected.

Every real vein raster is drawn from a drainage tree (``tree.drainage_tree``), so its veins are one
piece; VeinFlow sometimes breaks the midrib or leaves segments detached. Measures per image
(8-connectivity, inside the mask):

- ``lcc_coarse``: share of the coarse vein pixels (value > ``COARSE``: midrib, secondary,
  tertiary) in the largest connected piece (1 for every real raster);
- ``lcc_all``: the same for every vein pixel (> ``metrics.VEIN``, finer veins included);
- ``pieces``: coarse pieces of at least ``MIN_PIECE`` px;
- ``midrib_lcc``: share of the midrib pixels (> ``metrics.MIDRIB``) in their largest piece.

Repairs of a drawing (no training), each returning a float image in [0, 1]:

- ``prune``: drop every vein piece not connected to the main one (the largest coarse piece);
- ``bridge``: join every detached piece to the main one along the cheapest path over the drawing
  (cost ``1 / (eps + v)^2``, as ``tree.drainage_tree``), drawn at the order of the piece it joins;
  the drawing is otherwise unchanged;
- ``retree``: rebuild the whole drawing as the real rasters are made: drainage tree from the base
  over the drawing, orders from the flow, ``raster.vein_image``;
- ``loopcut``: cut every loop where two tree paths meet;
- ``bridge_loopcut`` (the one used): ``loopcut``, then ``bridge``.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra

from leafgen.veins import metrics, raster
from leafgen.veins.tree import _pixel_graph

COARSE, MIN_PIECE = 0.4, 4
EIGHT = np.ones((3, 3), bool)


def _float(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32) / (255.0 if v.dtype == np.uint8 else 1.0)


def _lcc(binary: np.ndarray) -> tuple[float, np.ndarray]:
    lab, n = ndi.label(binary, EIGHT)
    if n == 0:
        return 1.0, np.zeros(0, int)
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / sizes.sum()), sizes


def connectivity(veins: np.ndarray, mask: np.ndarray) -> dict:
    v = _float(veins) * (mask > 0)
    lcc_coarse, sizes = _lcc(v > COARSE)
    return {
        "lcc_coarse": lcc_coarse,
        "lcc_all": _lcc(v > metrics.VEIN)[0],
        "pieces": int((sizes >= MIN_PIECE).sum()),
        "midrib_lcc": _lcc(v > metrics.MIDRIB)[0],
    }


def summary(images: np.ndarray, masks: np.ndarray) -> dict:
    rows = [connectivity(v, m) for v, m in zip(images, masks, strict=True)]
    out = {}
    for k in rows[0]:
        a = np.array([r[k] for r in rows], float)
        out[f"{k}_median"] = float(np.median(a))
        out[f"{k}_mean"] = float(a.mean())
    a = np.array([r["lcc_coarse"] for r in rows])
    out["share_connected"] = float((a > 0.999).mean())  # coarse veins in one piece
    out["share_below_0.8"] = float((a < 0.8).mean())
    out["midrib_whole"] = float((np.array([r["midrib_lcc"] for r in rows]) > 0.999).mean())
    return out


def _main_piece(v: np.ndarray, threshold: float = COARSE) -> np.ndarray:
    lab, n = ndi.label(v > threshold, EIGHT)
    if n == 0:
        return np.zeros(v.shape, bool)
    return lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))


def prune(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = _float(veins) * (mask > 0)
    lab, _ = ndi.label(v > metrics.VEIN, EIGHT)
    main = _main_piece(v)
    keep = np.isin(lab, np.unique(lab[main]))
    return v * keep


def _order(value: float) -> int:
    """0 midrib, 1 secondary, 2 tertiary, 3 finer, from a drawn value (``metrics`` thresholds)."""
    return int(np.select([value > metrics.MIDRIB, value > metrics.SECONDARY,
                          value >= metrics.FINE_TOP], [0, 1, 2], 3))


def _bridge(v: np.ndarray, m: np.ndarray, threshold: float, eps: float, alpha: float) -> np.ndarray:
    main = _main_piece(v, max(threshold, COARSE))
    if not main.any():
        return v
    lab, n = ndi.label(v > threshold, EIGHT)
    attached = set(np.unique(lab[main])) - {0}
    loose = [k for k in range(1, n + 1) if k not in attached]
    if not loose:
        return v
    points, index, graph = _pixel_graph(m, 1.0 / (eps + v) ** alpha)
    distance, pred, _ = dijkstra(
        graph, indices=index[main & m], min_only=True, return_predecessors=True
    )
    size = v.shape[0]
    out = v.copy()
    piece = lab[tuple(points.T)]
    for k in loose:
        nodes = np.flatnonzero(piece == k)
        if not len(nodes):
            continue
        node = nodes[np.argmin(distance[nodes])]
        o = _order(float(np.percentile(v[lab == k], 90)))
        if threshold >= metrics.MIDRIB:  # midrib pieces joined as midrib
            if (lab == k).sum() < MIN_PIECE:
                continue
            o = 0
        elif threshold >= COARSE:  # a coarse piece gets a coarse bridge
            o = min(o, 2)
        path = []
        while node >= 0 and not main[tuple(points[node])]:
            path.append(points[node])
            node = pred[node]
        if node >= 0:
            path.append(points[node])
        layer = np.zeros_like(out)
        width = max(1, round(raster.WIDTH[o] * size))
        line = np.asarray(path)[:, ::-1].astype(np.int32)
        cv2.polylines(layer, [line], False, raster.VALUE[o], width)
        out = np.maximum(out, layer * m)
    return out


def bridge(veins: np.ndarray, mask: np.ndarray, eps: float = 0.05, alpha: float = 2.0) -> np.ndarray:
    """First the midrib pieces (> ``metrics.MIDRIB``, at least ``MIN_PIECE`` px) are joined to the
    largest one, as midrib. Then every coarse piece (> ``COARSE``) not in the main piece is joined to it by the cheapest path
    from the main piece (one multi-source Dijkstra), drawn with the width and value of the piece's
    order (from its 90th percentile value); then the same for the remaining pieces of finer veins,
    so that, as in a drainage tree, the coarse veins are one piece by themselves."""
    v = _float(veins) * (mask > 0)
    m = mask > 0
    for threshold in (metrics.MIDRIB, COARSE, metrics.VEIN):
        v = _bridge(v, m, threshold, eps, alpha)
    return v


def retree(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The drawing's drainage tree (``leaf.raster_tree`` without dropping the bridges),
    drawn back as the real rasters."""
    from skimage.morphology import skeletonize

    from leafgen.veins.raster import BOX_X, BOX_Y
    from leafgen.veins.tree import VeinTree, drainage_tree, nearest_inside

    v = _float(veins) * (mask > 0)
    m = mask > 0
    size = v.shape[0]
    sinks = skeletonize(v > metrics.VEIN)

    def rc(x, y):
        col = (x - BOX_X[0]) / (BOX_X[1] - BOX_X[0]) * size
        row = (y - BOX_Y[0]) / (BOX_Y[1] - BOX_Y[0]) * size
        return nearest_inside(m, np.array([col, row]))

    midrib = cv2.GaussianBlur((v > metrics.MIDRIB).astype(np.float32), (0, 0), 1)
    tree = drainage_tree(
        np.clip(v / 0.45, 0, 1), m, rc(0, 0), sinks, tip_rc=rc(0, 1), midrib_map=v + midrib,
        taper_px=1.0,
    )
    span = np.array([BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0]])
    tree = VeinTree(tree.child, tree.parent, tree.flow).mapped(
        lambda p: np.asarray(p, float) * span / size + [BOX_X[0], BOX_Y[0]]
    )
    return raster.vein_image(tree, size, fine=True) * m


def _base_rc(m: np.ndarray, on: np.ndarray) -> int:
    """Index (into ``np.argwhere(on)``) of the pixel of ``on`` nearest the leaflet base."""
    size = m.shape[0]
    r = (0 - raster.BOX_Y[0]) / (raster.BOX_Y[1] - raster.BOX_Y[0]) * size
    c = (0 - raster.BOX_X[0]) / (raster.BOX_X[1] - raster.BOX_X[0]) * size
    p = np.argwhere(on)
    return int(np.argmin((p[:, 0] - r) ** 2 + (p[:, 1] - c) ** 2))


def loopcut(
    veins: np.ndarray, mask: np.ndarray, min_loop: float = 12.0, top: float = metrics.SECONDARY
) -> np.ndarray:
    """Open the closed loops of a drawing as the real rasters are open (drainage trees): on the
    skeleton of the veins, the shortest-path tree from the base (cost ``1 / (eps + v)^2``); each
    skeleton edge outside the tree closes a loop, which is cut there (where the paths from the
    two sides meet, as in the real extraction) by erasing the 3 x 3 pixels around its two ends,
    when the loop is longer than ``min_loop`` px (smaller ones are skeleton artefacts) and the
    veins there are tertiary or finer (value <= ``top``)."""
    from skimage.morphology import skeletonize

    v = _float(veins) * (mask > 0)
    skel = skeletonize(v > metrics.VEIN)
    if skel.sum() < 2:
        return v
    points, index, graph = _pixel_graph(skel, 1.0 / (0.05 + v) ** 2)
    distance, pred = dijkstra(graph, indices=_base_rc(mask > 0, skel), return_predecessors=True)
    hops = dijkstra(_pixel_graph(skel, np.ones_like(v))[2], indices=_base_rc(mask > 0, skel))
    g = graph.tocoo()
    out = v.copy()
    for a, b in zip(g.row, g.col, strict=True):
        if a >= b or pred[a] == b or pred[b] == a or not np.isfinite(distance[a] + distance[b]):
            continue
        # loop length: the two tree paths up to their common ancestor
        seen, x = {}, a
        while x >= 0:
            seen[x] = hops[x]
            x = pred[x]
        y = b
        while y >= 0 and y not in seen:
            y = pred[y]
        if y < 0 or hops[a] + hops[b] - 2 * hops[y] < min_loop:
            continue
        ra, rb = points[a], points[b]
        if max(v[tuple(ra)], v[tuple(rb)]) > top:
            continue
        for r, c in (ra, rb):
            out[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2] = 0
    return out


def bridge_loopcut(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return bridge(loopcut(veins, mask), mask)
