"""Leaf-architecture statistics of the midrib and the secondaries, and a copying check.

The 16 statistics of ``veins.metrics`` are measured on the image (densities, gaps, orientation
histograms); here the midrib and each secondary are traced as paths, so their shape can be
measured with the characters botanists use to describe venation (Ellis et al., *Manual of Leaf
Architecture*, 2009, chapter "Vein characters"):

- ``sec_angle``: mean major secondary angle to the midvein, measured as in the Manual (char. 32):
  vertex at the junction, one ray along the midvein distally, the other along the secondary for
  25 % of its length; secondaries with their junction below 0.75 of the midrib;
- ``sec_angle_sd``: its standard deviation (the Manual's "uniform" vs "inconsistent", 10 deg);
- ``sec_spacing_cv``: coefficient of variation of the distance between adjacent secondaries along
  the midvein, per side (char. 31, regular vs irregular spacing);
- ``sec_per_length``: secondaries per unit midrib length (box units);
- ``sec_tort``, ``sec_tort_p90``: tortuosity (path length / chord) of the secondaries, mean and
  90th percentile (the "bizarre paths"); ``mid_tort``, ``mid_dev``: tortuosity of the midrib and
  its largest distance from its chord / chord. Tortuosity as arc / chord is the "distance metric"
  of vessel analysis (adapted here; not a Manual character);
- ``sec_reach``: mean distance from the end of each secondary to the outline (box units):
  do the secondaries run to the margin (craspedodromous, char. 27.1) or stop short;
- ``sec_length``: mean secondary length / midrib length.

Paths come from the image itself, the same way for real rasters and drawings: midrib = largest
piece of pixels > ``metrics.MIDRIB``, its skeleton's longest path from the base; a secondary =
a piece of pixels > ``metrics.SECONDARY`` outside the midrib that touches it, traced on its
skeleton from the pixel nearest the midrib to the farthest one. A drawing whose secondary is
detached from the midrib loses that secondary (connectivity is measured by ``veins.connect``).

``copying``: for each outline, the chamfer distance between its coarse veins and those of the
``K`` training leaflets with the most similar outline (IoU), the smallest one. Real development
leaflets give the reference: a generator that copies its training set is much closer to its
nearest training leaflet than a real leaflet is.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from scipy.stats import wasserstein_distance

from leafgen.veins import connect, metrics
from leafgen.veins.tree import _pixel_graph

NAMES = (
    "mid_tort",
    "mid_dev",
    "sec_per_length",
    "sec_angle",
    "sec_angle_sd",
    "sec_spacing_cv",
    "sec_tort",
    "sec_tort_p90",
    "sec_reach",
    "sec_length",
)
EIGHT = np.ones((3, 3), bool)
K = 20


def _longest_path(skel: np.ndarray, start_rc: np.ndarray) -> np.ndarray:
    """(n, 2) rows/cols of the longest geodesic path on ``skel`` from its pixel nearest
    ``start_rc``."""
    points, index, graph = _pixel_graph(skel, np.ones(skel.shape))
    s = int(np.argmin(((points - start_rc) ** 2).sum(1)))
    dist, pred = dijkstra(graph, indices=s, return_predecessors=True)
    dist[~np.isfinite(dist)] = -1
    node, path = int(np.argmax(dist)), []
    while node >= 0:
        path.append(points[node])
        node = pred[node]
    return np.asarray(path[::-1], float)


def _length(path: np.ndarray) -> float:
    return float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0


def statistics(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    v = connect._float(veins) * (mask > 0)
    m = mask > 0
    size = v.shape[0]
    out = np.full(len(NAMES), np.nan)
    lab, n = ndi.label(v > metrics.MIDRIB, EIGHT)
    if n == 0:
        return out
    midrib_all = lab > 0
    midrib = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    base = np.array([metrics._y_px(0.0, size), metrics._x_px(0.0, size)], float)
    mid = _longest_path(skeletonize(midrib), base)
    if len(mid) < 10:
        return out
    if np.hypot(*(mid[0] - base)) > np.hypot(*(mid[-1] - base)):
        mid = mid[::-1]
    lm = _length(mid)
    chord = mid[-1] - mid[0]
    cl = float(np.hypot(*chord))
    rel = mid - mid[0]
    dev = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / max(cl, 1e-9)
    out[0], out[1] = lm / max(cl, 1e-9), dev.max() / max(cl, 1e-9)
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(mid, axis=0).T))] / max(lm, 1e-9)

    near = ndi.binary_dilation(midrib_all, iterations=4)
    sec = (v > metrics.SECONDARY) & ~ndi.binary_dilation(midrib_all, iterations=2) & m
    slab, sn = ndi.label(sec, EIGHT)
    edge = ndi.distance_transform_edt(m) / size
    angles, pos, side, tort, reach, length = [], [], [], [], [], []
    for k in range(1, sn + 1):
        piece = slab == k
        if piece.sum() < 8 or not (piece & near).any():
            continue
        skel = skeletonize(piece)
        if skel.sum() < 6:
            continue
        sk = np.argwhere(skel)
        d = ((sk[:, None, :] - mid[None, ::2, :]) ** 2).sum(-1)
        j = int(np.argmin(d.min(1)))
        path = _longest_path(skel, sk[j])
        ls = _length(path)
        if ls < 0.03 * size:
            continue
        i = int(np.argmin(((mid - path[0]) ** 2).sum(1)))
        t = mid[min(i + 5, len(mid) - 1)] - mid[max(i - 5, 0)]  # midvein, distally
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        a = path[int(np.searchsorted(cum, 0.25 * ls))] - path[0]
        cos = a @ t / (np.linalg.norm(a) * np.linalg.norm(t) + 1e-9)
        if arc[i] < 0.75:
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        rel = path[0] - mid[i]
        pos.append(arc[i])
        side.append(np.sign(t[0] * rel[1] - t[1] * rel[0]))
        tort.append(ls / max(np.hypot(*(path[-1] - path[0])), 1e-9))
        r, c = np.round(path[-1]).astype(int)
        reach.append(edge[r, c])
        length.append(ls / lm)
    if len(tort) < 2:
        return out
    pos, side = np.asarray(pos), np.asarray(side)
    gaps = np.concatenate([np.diff(np.sort(pos[side == s])) for s in (-1, 1)])
    gaps = gaps[gaps > 0.01]  # the two branches of one forked secondary
    out[2] = len(tort) / (lm / size * (metrics.BOX_Y[1] - metrics.BOX_Y[0]))
    out[3] = np.mean(angles) if angles else np.nan
    out[4] = np.std(angles) if len(angles) > 1 else np.nan
    out[5] = gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan
    out[6], out[7] = np.mean(tort), np.percentile(tort, 90)
    out[8], out[9] = np.mean(reach), np.mean(length)
    return out


def table(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.array([statistics(v, m) for v, m in zip(images, masks, strict=True)])


def compare(gen: np.ndarray, real: np.ndarray, names: tuple = NAMES) -> dict:
    """W1 / real std per statistic (rows with that statistic), 1-NNA on complete rows."""
    out = {}
    for i, k in enumerate(names):
        a, b = gen[:, i], real[:, i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) == 0:  # e.g. a drawing without secondaries
            out[f"w1_{k}"] = out[f"mean_{k}"] = np.nan
            continue
        out[f"w1_{k}"] = float(wasserstein_distance(a, b) / (b.std() + 1e-9))
        out[f"mean_{k}"] = float(a.mean())
    ok_g, ok_r = np.isfinite(gen).all(1), np.isfinite(real).all(1)
    mu, sd = real[ok_r].mean(0), real[ok_r].std(0) + 1e-9
    out["one_nna"] = (
        metrics._one_nna((gen[ok_g] - mu) / sd, (real[ok_r] - mu) / sd) if ok_g.any() else np.nan
    )
    out["w1_mean"] = float(np.mean([out[f"w1_{k}"] for k in names]))
    out["complete"] = float(ok_g.mean())
    return out


def _coarse_dt(v: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(~(connect._float(v) > connect.COARSE))


def copying(query: np.ndarray, masks: np.ndarray, train_v: np.ndarray, train_m: np.ndarray):
    """Per query image: smallest symmetric chamfer distance (px, coarse veins) to the ``K``
    training leaflets with the highest outline IoU (masks compared at 64 px)."""
    small = lambda a: a[:, ::4, ::4].reshape(len(a), -1).astype(np.float32)
    tm, qm = small(train_m > 0), small(masks > 0)
    inter = qm @ tm.T
    iou = inter / (qm.sum(1)[:, None] + tm.sum(1) - inter)
    out = []
    for q, row in zip(query, iou, strict=True):
        qv = connect._float(q) > connect.COARSE
        qdt = ndi.distance_transform_edt(~qv)
        best = np.inf
        for j in np.argsort(-row)[:K]:
            tv = connect._float(train_v[j]) > connect.COARSE
            if not qv.any() or not tv.any():
                continue
            d = 0.5 * (qdt[tv].mean() + ndi.distance_transform_edt(~tv)[qv].mean())
            best = min(best, d)
        out.append(best)
    return np.asarray(out)
