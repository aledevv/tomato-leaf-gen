"""Drainage tree: the shortest-path tree from the leaflet base over a cost that is cheap on veins.

Every centerline pixel sends one unit of flow to the base along its path; the flow a pixel
carries ranks it like a river network (midrib > secondary > tertiary). The result is always a
tree, so closed loops of the real tertiary network are cut, and flow measures drained area, not
vein width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class VeinTree:
    """Edges child -> parent as two (n, 2) xy arrays plus the flow of each child."""

    child: np.ndarray
    parent: np.ndarray
    flow: np.ndarray

    def pruned(self, min_flow: float) -> VeinTree:
        keep = self.flow >= min_flow
        return VeinTree(self.child[keep], self.parent[keep], self.flow[keep])

    def mapped(self, fn) -> VeinTree:
        return VeinTree(fn(self.child), fn(self.parent), self.flow)

    def save(self, path: Path, **extra) -> None:
        np.savez_compressed(path, child=self.child, parent=self.parent, flow=self.flow, **extra)

    @classmethod
    def load(cls, path: Path) -> VeinTree:
        d = np.load(path)
        return cls(d["child"], d["parent"], d["flow"])


def _pixel_graph(mask: np.ndarray, cost: np.ndarray):
    index = -np.ones(mask.shape, np.int64)
    points = np.argwhere(mask)
    index[tuple(points.T)] = np.arange(len(points))
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        q = points + [dy, dx]
        ok = (q[:, 0] < mask.shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < mask.shape[1])
        j = -np.ones(len(points), np.int64)
        j[ok] = index[q[ok, 0], q[ok, 1]]
        a, b = np.nonzero(j >= 0)[0], j[j >= 0]
        w = np.hypot(dy, dx) * 0.5 * (cost[tuple(points[a].T)] + cost[tuple(points[b].T)])
        rows += [a, b]
        cols += [b, a]
        weights += [w, w]
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(points),) * 2,
    ).tocsr()
    return points, index, graph


def _midrib(mask, cost, base_rc, tip_rc, axis_width, smooth):
    """Cheapest base -> tip path under the axis prior, smoothed along its length (Gaussian,
    ``smooth`` of the base-tip distance) and drawn back as a 1 px line inside the mask."""
    rr, cc = np.indices(mask.shape)
    b, t = base_rc.astype(float), tip_rc.astype(float)
    d = t - b
    length = float(np.hypot(*d))
    s = np.clip(((rr - b[0]) * d[0] + (cc - b[1]) * d[1]) / length**2, 0, 1)
    off = np.hypot(rr - b[0] - s * d[0], cc - b[1] - s * d[1]) / (axis_width * length)
    points, index, graph = _pixel_graph(mask, cost * (1 + off**2))
    _, pred = dijkstra(graph, indices=index[tuple(base_rc)], return_predecessors=True)
    path, node = [], index[tuple(tip_rc)]
    while node >= 0:
        path.append(points[node])
        node = pred[node]
    path = np.asarray(path, float)
    if smooth > 0 and len(path) > 2:
        path = gaussian_filter1d(path, smooth * length, axis=0, mode="nearest")
    line = np.zeros(mask.shape, np.uint8)
    cv2.polylines(line, [np.round(path[:, ::-1]).astype(np.int32)], False, 1, 1)
    return (line > 0) & mask


def drainage_tree(
    vmap: np.ndarray,
    mask: np.ndarray,
    base_rc: np.ndarray,
    sinks: np.ndarray,
    *,
    tip_rc: np.ndarray | None = None,
    midrib_map: np.ndarray | None = None,
    alpha: float = 2.0,
    eps: float = 0.05,
    taper_px: float = 12.0,
    min_flow: float = 3.0,
    axis_width: float = 0.025,
    midrib_smooth: float = 0.03,
    midrib_cost: float = 0.02,
) -> VeinTree:
    """Tree in pixel xy. ``base_rc`` is (row, col) inside ``mask``; ``sinks`` a boolean image.
    The map fades to zero within ``taper_px`` of the outline, because the shaded leaf edge is a
    ridge too and paths would run along it.

    With ``tip_rc`` the midrib is traced first, as the cheapest base -> tip path over a cost that
    grows with the distance from the base-tip segment (``axis_width`` of the leaf length), so a
    fold, an overlapping leaflet or a dark photo cannot pull it sideways. ``midrib_map`` (default
    ``vmap``) is the map used for that path: the midrib is wider than the other veins, so a
    coarser-scale map helps. The path is smoothed, its pixels become almost free
    (``midrib_cost``) and every other vein drains into it."""
    taper = np.clip(distance_transform_edt(mask) / taper_px, 0, 1)
    vmap = vmap * taper
    cost = 1.0 / (eps + vmap) ** alpha
    if tip_rc is not None:
        mcost = cost if midrib_map is None else 1.0 / (eps + midrib_map * taper) ** alpha
        midrib = _midrib(
            mask, mcost, np.asarray(base_rc), np.asarray(tip_rc), axis_width, midrib_smooth
        )
        cost = np.where(midrib, cost.min() * midrib_cost, cost)
    points, index, graph = _pixel_graph(mask, cost)
    distance, pred = dijkstra(graph, indices=index[tuple(base_rc)], return_predecessors=True)
    flow = np.zeros(len(points))
    s = index[sinks & mask]
    flow[s] = 1.0
    for node in np.argsort(-distance):  # farthest first, so children are done before parents
        if pred[node] >= 0:
            flow[pred[node]] += flow[node]
    keep = (flow >= min_flow) & (pred >= 0)
    return VeinTree(
        points[keep][:, ::-1].astype(float), points[pred[keep]][:, ::-1].astype(float), flow[keep]
    )


def nearest_inside(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(row, col) of the pixel at image point ``xy`` (col, row), or of the nearest pixel of the mask
    when that one lies outside it."""
    r = int(np.clip(round(xy[1]), 0, mask.shape[0] - 1))
    c = int(np.clip(round(xy[0]), 0, mask.shape[1] - 1))
    if not mask[r, c]:
        rr, cc = np.nonzero(mask)
        k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
        r, c = int(rr[k]), int(cc[k])
    return np.array([r, c])
