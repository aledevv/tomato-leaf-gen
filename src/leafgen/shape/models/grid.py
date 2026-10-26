"""The canonical box shared by the raster shape models (x in [-0.6, 0.6], y in [-0.1, 1.1]) and the
contour of a field drawn on it: largest zero isoline, re-canonicalised on its base and tip."""

from __future__ import annotations

import numpy as np

from leafgen.shape.canonicalize import canonical_contour
from leafgen.shape.features import valid_contour

GRID, BOX_X, BOX_Y, SIZE = 128, (-0.6, 0.6), (-0.1, 1.1), 1.2


def contour_from_sdf(field: np.ndarray) -> np.ndarray | None:
    """Largest zero isoline of an (n, n) field -> canonical 1024-point contour, or None."""
    from skimage.measure import find_contours

    n = field.shape[0]
    loops = [c for c in find_contours(field, 0.0) if len(c) > 16 and np.allclose(c[0], c[-1])]
    if not loops:
        return None
    rc = max(loops, key=len)[:-1]
    xy = np.column_stack(
        [BOX_X[0] + (rc[:, 1] + 0.5) * SIZE / n, BOX_Y[0] + (rc[:, 0] + 0.5) * SIZE / n]
    )
    base = xy[int(np.argmin(np.linalg.norm(xy, axis=1)))]
    tip = xy[int(np.argmin(np.linalg.norm(xy - [0.0, 1.0], axis=1)))]
    try:
        contour = canonical_contour(xy, base, tip)
    except ValueError:
        return None
    return contour if valid_contour(contour) else None
