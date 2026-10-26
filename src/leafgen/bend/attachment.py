"""Where the biological attachment (0, 0) sits on the flat semantic mesh."""

from __future__ import annotations

import numpy as np

from leafgen.contracts.geometry import SemanticFlatMesh2D


class AttachmentOutsideSurfaceError(ValueError):
    """The biological attachment is not represented by the flat mesh."""


def interpolate_margin_at_point(
    semantic: SemanticFlatMesh2D,
    point_xy: np.ndarray,
    *,
    tolerance: float = 1e-10,
    snap_distance: float = 0.01,
) -> float:
    """Barycentrically interpolate margin_v at an interior or boundary point; a point at most
    ``snap_distance`` outside the mesh is moved to the closest point of the mesh."""
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("interpolation point must be a finite 2-vector")
    candidates: list[float] = []
    vertices = semantic.mesh.vertices_xy
    for face in np.asarray(semantic.mesh.faces, dtype=np.int64):
        triangle = vertices[face]
        first = triangle[1] - triangle[0]
        second = triangle[2] - triangle[0]
        determinant = first[0] * second[1] - first[1] * second[0]
        if abs(float(determinant)) <= 1e-15:
            raise ValueError("cannot interpolate on a degenerate triangle")
        relative = point - triangle[0]
        w1 = (relative[0] * second[1] - relative[1] * second[0]) / determinant
        w2 = (first[0] * relative[1] - first[1] * relative[0]) / determinant
        weights = np.asarray([1.0 - w1 - w2, w1, w2])
        if np.all(weights >= -tolerance) and np.all(weights <= 1.0 + tolerance):
            weights /= np.sum(weights)
            candidates.append(float(weights @ semantic.margin_v[face]))
    if not candidates:  # just outside the boundary: move it to the closest point of the mesh
        faces = np.asarray(semantic.mesh.faces, dtype=np.int64)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
        along = np.clip(np.sum((point - a) * (b - a), axis=1) / np.sum((b - a) ** 2, axis=1), 0, 1)
        distance = np.linalg.norm(a + along[:, None] * (b - a) - point, axis=1)
        k = int(np.argmin(distance))
        if distance[k] > snap_distance:
            raise AttachmentOutsideSurfaceError("biological attachment is outside the flat mesh")
        v = semantic.margin_v
        return float((1 - along[k]) * v[edges[k, 0]] + along[k] * v[edges[k, 1]])
    if float(np.ptp(candidates)) > max(10.0 * tolerance, 1e-10):
        raise ValueError("attachment-field interpolation is ambiguous")
    return candidates[0]
