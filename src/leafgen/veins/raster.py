"""Vein trees as images in the frame of the mask models (``shape.models.grid``).

The mask models draw the outline in the box x in [-0.6, 0.6], y in [-0.1, 1.1] of the shape
canonical frame (``shape.canonicalize``: base (0, 0), tip (0, 1), counter-clockwise). The vein
stage maps the photo with y flipped first, so its x is mirrored with respect to that frame:
``to_shape_frame`` undoes it. A tree becomes one channel: anti-aliased lines whose width and value
grow with the vein order (midrib, secondary, tertiary); the finer veins are left out, or drawn 1 px
wide at value 0.35 with ``--fine`` (meant for 256 px, where they are resolved). The outline
mask is the one the mask models train on (EFD h60 of the bank -> pseudo-landmarks -> contour), and
the veins are clipped to it; ``iou`` records how well the photo outline of the vein stage matches it.
"""

from __future__ import annotations

import cv2
import numpy as np

from leafgen.veins.tree import VeinTree

BOX_X, BOX_Y = (-0.6, 0.6), (-0.1, 1.1)
SECONDARY, TERTIARY = 0.04, 0.008  # flow / max flow
# width as a fraction of the box side (at least 1 px), and value, per order: midrib, secondary,
# tertiary, finer
WIDTH = (0.012, 0.007, 0.004, 0.0)
VALUE = (1.0, 0.75, 0.45, 0.35)


def to_shape_frame(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) * [-1.0, 1.0]


def orders(tree: VeinTree) -> np.ndarray:
    """0 midrib (the path from the node nearest the tip to the base), 1 secondary, 2 tertiary,
    3 finer, per edge."""
    f = tree.flow / tree.flow.max()
    k = np.select([f > SECONDARY, f > TERTIARY], [1, 2], 3)
    key = {(round(float(a[0]), 9), round(float(a[1]), 9)): i for i, a in enumerate(tree.child)}
    i = int(np.argmin(np.hypot(tree.child[:, 0], tree.child[:, 1] - 1)))
    seen = set()
    while i is not None and i not in seen:
        seen.add(i)
        k[i] = 0
        p = tree.parent[i]
        i = key.get((round(float(p[0]), 9), round(float(p[1]), 9)))
    return k


def _px(points: np.ndarray, size: int) -> np.ndarray:
    span = np.array([BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0]])
    return (np.asarray(points) - [BOX_X[0], BOX_Y[0]]) * (size / span)


def vein_image(
    tree: VeinTree, size: int, order: np.ndarray | None = None, fine: bool = False
) -> np.ndarray:
    """float32 (size, size) in [0, 1]; ``tree`` already in the shape frame; ``fine`` also draws
    the finer veins."""
    if len(tree.child) == 0:
        return np.zeros((size, size), np.float32)
    order = orders(tree) if order is None else order
    sub = 8  # sub-pixel bits of cv2.line
    img = np.zeros((size, size), np.float32)
    c, p = _px(tree.child, size), _px(tree.parent, size)
    for k in (3, 2, 1, 0) if fine else (2, 1, 0):  # finer first, so the midrib is on top
        width = max(1, round(WIDTH[k] * size))
        layer = np.zeros_like(img)
        for a, b in zip(c[order == k], p[order == k], strict=True):
            cv2.line(
                layer,
                tuple(np.round(a * 2**sub).astype(int)),
                tuple(np.round(b * 2**sub).astype(int)),
                VALUE[k],
                width,
                cv2.LINE_AA,
                sub,
            )
        img = np.maximum(img, layer)
    return img


def order_layers(tree: VeinTree, size: int, fine: bool = False) -> np.ndarray:
    """float32 (3 or 4, size, size) in [0, 1]: midrib, secondary, tertiary (and finer) each on its
    own channel, drawn as in ``vein_image`` (same widths, anti-aliasing), so that
    ``vein_image == max_k VALUE[k] * layer_k`` up to rounding."""
    k_max = 4 if fine else 3
    out = np.zeros((k_max, size, size), np.float32)
    if len(tree.child) == 0:
        return out
    order = orders(tree)
    sub = 8
    c, p = _px(tree.child, size), _px(tree.parent, size)
    for k in range(k_max):
        width = max(1, round(WIDTH[k] * size))
        for a, b in zip(c[order == k], p[order == k], strict=True):
            cv2.line(
                out[k],
                tuple(np.round(a * 2**sub).astype(int)),
                tuple(np.round(b * 2**sub).astype(int)),
                1.0,
                width,
                cv2.LINE_AA,
                sub,
            )
    return np.clip(out, 0, 1)


def compose(layers: np.ndarray) -> np.ndarray:
    """(..., k, H, W) per-order layers in [0, 1] -> (..., H, W) one-channel image as ``vein_image``."""
    values = np.asarray(VALUE[: layers.shape[-3]], np.float32).reshape(-1, 1, 1)
    return (layers * values).max(-3)


def outline_mask(contour: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [np.round(_px(contour, size) * 16).astype(np.int32)], 1, shift=4)
    return mask
