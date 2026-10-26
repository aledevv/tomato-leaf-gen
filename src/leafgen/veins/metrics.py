"""Statistics of vein images and distances between two sets of them.

Works on the images of ``veins.raster`` (value 1 midrib, 0.75 secondary, 0.45 tertiary, inside the
outline mask), so real trees, baselines (donor transfer, space colonisation) and model outputs go
through the same measurements. Per image, a vector of:

- ``density``: vein pixels / leaf pixels;
- ``gap_mean``, ``gap_p90``: distance from each leaf pixel to the nearest vein, as a fraction of
  the box side (areole size);
- ``midrib``: fraction of the base-tip span crossed by a midrib-valued pixel near x = 0;
- ``secondaries``: secondary crossings of two lines parallel to the axis (x = +-0.12), per unit
  length of axis (spacing);
- ``tertiary``: share of tertiary pixels among vein pixels;
- ``lobe_reach``: mean distance from the lobe tips (sharp convex-hull vertices of the outline,
  away from base and tip) to the nearest secondary-or-larger vein, as a fraction of the box side:
  do the veins go where the outline has lobes;
- ``angle_0`` .. ``angle_5``: histogram of the orientation of non-midrib vein pixels with respect
  to the axis (structure tensor), 15 degree bins;
- with ``fine`` (images with the finer veins, ``raster.vein_image(..., fine=True)``), also ``fine_share`` (share of
  vein pixels in the finer band, 0.3-0.42), ``areoles`` (vein-free regions of at least 4 px per
  1000 leaf pixels) and ``areole_median`` (their median area per 1000 leaf pixels).

``binary_statistics``: only the statistics that need no vein order (``BINARY``: density, gaps,
areoles, orientation of all veins), on the image thresholded at ``VEIN``; for comparing with
generators that draw veins as one binary class (``veins.benfenati``).

Set distances: per-statistic Wasserstein-1 divided by the real standard deviation, and 1-NNA of
the standardised vectors (0.5 = indistinguishable). ``ideal`` gives both for real against real
(two halves split by photo), the floor a generator can reach.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from leafgen.veins.raster import BOX_X, BOX_Y

NAMES = (
    "density",
    "gap_mean",
    "gap_p90",
    "midrib",
    "secondaries",
    "tertiary",
    "lobe_reach",
    *(f"angle_{i}" for i in range(6)),
)
FINE = ("fine_share", "areoles", "areole_median")
BINARY = (
    "density",
    "gap_mean",
    "gap_p90",
    "areoles",
    "areole_median",
    *(f"angle_{i}" for i in range(6)),
)
VEIN, SECONDARY, MIDRIB, FINE_TOP = 0.3, 0.6, 0.9, 0.42


def _x_px(x: float, size: int) -> int:
    return round((x - BOX_X[0]) / (BOX_X[1] - BOX_X[0]) * size)


def _y_px(y: float, size: int) -> int:
    return round((y - BOX_Y[0]) / (BOX_Y[1] - BOX_Y[0]) * size)


def lobe_tips(mask: np.ndarray, sharp: float = 150.0) -> np.ndarray:
    """(row, col) of the vertices of the outline's convex hull (simplified to 1.5 % of the box)
    with an interior angle below ``sharp`` degrees, more than 0.1 of the axis from base and tip."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=len)[:, 0, :].astype(float)
    hull = cv2.convexHull(c.astype(np.float32))
    hull = cv2.approxPolyDP(hull, 0.015 * mask.shape[0], True)[:, 0, :].astype(float)
    if len(hull) < 3:
        return np.zeros((0, 2))
    size = mask.shape[0]
    ends = np.array([[_x_px(0, size), _y_px(y, size)] for y in (0.0, 1.0)], float)
    reach = 0.1 / (BOX_Y[1] - BOX_Y[0]) * size
    out = []
    for i, p in enumerate(hull):
        a, b = hull[i - 1] - p, hull[(i + 1) % len(hull)] - p
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if (
            np.degrees(np.arccos(np.clip(cos, -1, 1))) < sharp
            and np.hypot(*(ends - p).T).min() > reach
        ):
            out.append(p[::-1])
    return np.asarray(out).reshape(-1, 2)


def statistics(veins: np.ndarray, mask: np.ndarray, fine: bool = False) -> np.ndarray:
    """``veins`` float in [0, 1] (or uint8 0-255), ``mask`` bool, both (size, size)."""
    v = veins.astype(np.float32) / (255.0 if veins.dtype == np.uint8 else 1.0)
    m = mask.astype(bool)
    size = v.shape[0]
    v = v * m
    vein = v > VEIN
    leaf = max(int(m.sum()), 1)
    gap = ndi.distance_transform_edt(~vein)[m] / size if vein.any() else np.full(leaf, 1.0)
    y0, y1 = _y_px(0.0, size), _y_px(1.0, size)
    band = v[y0:y1, _x_px(-0.05, size) : _x_px(0.05, size) + 1]
    midrib = float((band > MIDRIB).any(axis=1).mean())
    crossings = 0
    for x in (-0.12, 0.12):
        line = v[_y_px(0.1, size) : _y_px(0.9, size), _x_px(x, size)] > SECONDARY
        crossings += int(np.sum(line[1:] & ~line[:-1]))
    n_vein = max(int(vein.sum()), 1)
    tertiary = float(((v > VEIN) & (v <= SECONDARY)).sum() / n_vein)
    tips = lobe_tips(m)
    major = ndi.distance_transform_edt(~(v > SECONDARY)) / size
    rc = np.clip(np.round(tips).astype(int), 0, size - 1)
    lobe_reach = float(major[rc[:, 0], rc[:, 1]].mean()) if len(rc) else 0.0
    gy, gx = np.gradient(ndi.gaussian_filter(v, 1.0))
    jxx, jyy, jxy = (ndi.gaussian_filter(a, 1.5) for a in (gx * gx, gy * gy, gx * gy))
    # line direction = perpendicular to the dominant gradient; angle to the axis (y) in [0, 90]
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    angle = np.degrees(np.abs(np.arctan2(np.cos(theta), np.abs(np.sin(theta)))))
    sel = vein & (v <= MIDRIB)
    hist = np.histogram(angle[sel], bins=6, range=(0, 90))[0].astype(float)
    hist /= max(hist.sum(), 1.0)
    extra = []
    if fine:
        areas = np.bincount(ndi.label(m & ~vein)[0].ravel())[1:]
        areas = areas[areas >= 4] / leaf * 1e3
        extra = [
            ((v > VEIN) & (v < FINE_TOP)).sum() / n_vein,
            len(areas) / leaf * 1e3,
            np.median(areas) if len(areas) else 0.0,
        ]
    return np.array(
        [
            vein.sum() / leaf,
            gap.mean(),
            np.percentile(gap, 90),
            midrib,
            crossings / 0.8,
            tertiary,
            lobe_reach,
            *hist,
            *extra,
        ]
    )


def binary_statistics(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``BINARY`` statistics of ``veins`` > ``VEIN`` (float in [0, 1] or uint8) inside ``mask``.
    Areoles are the regions closed by the veins and the 1 px leaf outline, as in a skeleton drawing
    (outline + veins) of the Benfenati baseline."""
    v = veins.astype(np.float32) / (255.0 if veins.dtype == np.uint8 else 1.0)
    m = mask.astype(bool)
    size = v.shape[0]
    vein = (v > VEIN) & m
    leaf = max(int(m.sum()), 1)
    gap = ndi.distance_transform_edt(~vein)[m] / size if vein.any() else np.full(leaf, 1.0)
    outline = m & ~ndi.binary_erosion(m, np.ones((3, 3), bool))
    areas = np.bincount(ndi.label(m & ~(vein | outline))[0].ravel())[1:]
    areas = areas[areas >= 4] / leaf * 1e3
    gy, gx = np.gradient(ndi.gaussian_filter(vein.astype(np.float32), 1.0))
    jxx, jyy, jxy = (ndi.gaussian_filter(a, 1.5) for a in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    angle = np.degrees(np.abs(np.arctan2(np.cos(theta), np.abs(np.sin(theta)))))
    hist = np.histogram(angle[vein], bins=6, range=(0, 90))[0].astype(float)
    hist /= max(hist.sum(), 1.0)
    return np.array(
        [
            vein.sum() / leaf,
            gap.mean(),
            np.percentile(gap, 90),
            len(areas) / leaf * 1e3,
            np.median(areas) if len(areas) else 0.0,
            *hist,
        ]
    )


def table(
    images: np.ndarray, masks: np.ndarray, fine: bool = False, binary: bool = False
) -> np.ndarray:
    if binary:
        return np.array([binary_statistics(v, m) for v, m in zip(images, masks, strict=True)])
    return np.array([statistics(v, m, fine) for v, m in zip(images, masks, strict=True)])


def _one_nna(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    x = np.vstack([a, b])
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    label = np.r_[np.zeros(n), np.ones(n)]
    return float((label[np.argmin(d, axis=1)] == label).mean())


def compare(generated: np.ndarray, real: np.ndarray) -> dict:
    """Statistic tables (rows = images): W1 / real std per statistic, its mean, and 1-NNA."""
    mu, sd = real.mean(0), real.std(0) + 1e-9
    names = BINARY if real.shape[1] == len(BINARY) else NAMES + FINE[: real.shape[1] - len(NAMES)]
    w = {k: wasserstein_distance(generated[:, i], real[:, i]) / sd[i] for i, k in enumerate(names)}
    return {
        "one_nna": _one_nna((generated - mu) / sd, (real - mu) / sd),
        "w1_mean": float(np.mean(list(w.values()))),
        **{f"w1_{k}": float(v) for k, v in w.items()},
    }


def ideal(real: np.ndarray, photo: np.ndarray, seed: int = 0) -> dict:
    """``compare`` between two halves of the real set, split by photo."""
    photos = np.unique(photo)
    half = set(np.random.default_rng(seed).permutation(photos)[: len(photos) // 2])
    a = np.array([p in half for p in photo])
    return compare(real[a], real[~a])
