"""Deterministic quality-constrained triangulation for leaf outlines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import triangle as triangle_backend

from leafgen.contracts.geometry import FlatMesh2D, LeafShape2D, signed_area


def _proper_segment_intersections(
    start: np.ndarray,
    end: np.ndarray,
    contour: np.ndarray,
    start_index: int,
    end_index: int,
    epsilon: float,
) -> bool:
    edge_start = contour
    edge_end = np.roll(contour, -1, axis=0)
    indices = np.arange(len(contour))
    excluded = (
        (indices == start_index)
        | ((indices + 1) % len(contour) == start_index)
        | (indices == end_index)
        | ((indices + 1) % len(contour) == end_index)
    )
    edge_start = edge_start[~excluded]
    edge_end = edge_end[~excluded]
    ab = end - start
    cross_a = ab[0] * (edge_start[:, 1] - start[1]) - ab[1] * (edge_start[:, 0] - start[0])
    cross_b = ab[0] * (edge_end[:, 1] - start[1]) - ab[1] * (edge_end[:, 0] - start[0])
    cd = edge_end - edge_start
    cross_c = cd[:, 0] * (start[1] - edge_start[:, 1]) - cd[:, 1] * (start[0] - edge_start[:, 0])
    cross_d = cd[:, 0] * (end[1] - edge_start[:, 1]) - cd[:, 1] * (end[0] - edge_start[:, 0])
    return bool(np.any((cross_a * cross_b < -(epsilon**2)) & (cross_c * cross_d < -(epsilon**2))))


def _forward_arc_indices(start: int, end: int, count: int) -> list[int]:
    indices = [start]
    current = start
    while current != end:
        current = (current + 1) % count
        indices.append(current)
    return indices


def _chord_is_inside(contour: np.ndarray, start_index: int, end_index: int, epsilon: float) -> bool:
    if (start_index + 1) % len(contour) == end_index:
        return True
    start = contour[start_index]
    end = contour[end_index]
    if _proper_segment_intersections(start, end, contour, start_index, end_index, epsilon):
        return False
    relative_start = contour - start
    relative_end = contour - end
    segment = end - start
    cross = np.abs(segment[0] * relative_start[:, 1] - segment[1] * relative_start[:, 0])
    dot = np.sum(relative_start * relative_end, axis=1)
    indices = np.arange(len(contour))
    if np.any(
        (indices != start_index) & (indices != end_index) & (cross <= epsilon) & (dot < -epsilon)
    ):
        return False
    midpoint = (start + end) * 0.5
    polygon = contour.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(polygon, tuple(midpoint), False) >= 0.0


def simplify_boundary_inside(
    contour: np.ndarray, *, tolerance: float, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify only chords proven to stay inside the original polygon."""
    points = np.asarray(contour, dtype=np.float64)
    if len(points) < 16 or signed_area(points) <= 0.0:
        raise ValueError("boundary simplification requires a CCW contour")
    proposed = cv2.approxPolyDP(
        points.astype(np.float32).reshape(-1, 1, 2), tolerance, True
    ).reshape(-1, 2)
    distances = np.linalg.norm(points[:, None, :] - proposed[None, :, :], axis=2)
    retained = sorted({int(index) for index in np.argmin(distances, axis=0)})
    if len(retained) < 3:
        raise ValueError("boundary simplification collapsed the polygon")
    final_indices: set[int] = set(retained)
    for start, end in zip(retained, [*retained[1:], retained[0]], strict=True):
        if not _chord_is_inside(points, start, end, epsilon):
            final_indices.update(_forward_arc_indices(start, end, len(points)))
    source_indices = np.asarray(sorted(final_indices), dtype=np.uint32)
    simplified = points[source_indices]
    if signed_area(simplified) <= 0.0:
        raise ValueError("boundary simplification changed contour orientation")
    return simplified, source_indices


def _canonical_faces(vertices: np.ndarray, faces: np.ndarray, epsilon: float) -> np.ndarray:
    result = np.asarray(faces, dtype=np.int64).copy()
    triangles = vertices[result]
    doubled = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    result[doubled < 0.0] = result[doubled < 0.0][:, ::-1]
    triangles = vertices[result]
    doubled = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    if np.any(doubled <= 2.0 * epsilon):
        raise ValueError("Triangle emitted a degenerate face")
    for row in result:
        row[:] = np.roll(row, -int(np.argmin(row)))
    order = np.lexsort((result[:, 2], result[:, 1], result[:, 0]))
    return result[order].astype(np.uint32)


def _canonical_edges(edges: np.ndarray) -> np.ndarray:
    result = np.sort(np.asarray(edges, dtype=np.uint32), axis=1)
    result = np.unique(result, axis=0)
    order = np.lexsort((result[:, 1], result[:, 0]))
    return result[order]


def _map_input_boundary(vertices: np.ndarray, boundary: np.ndarray, epsilon: float) -> np.ndarray:
    squared = np.sum((boundary[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(boundary)), indices])
    if np.any(distances > epsilon) or len(np.unique(indices)) != len(indices):
        raise ValueError("Triangle did not preserve every input boundary vertex")
    return indices.astype(np.uint32)


@dataclass(frozen=True)
class TriangleMeshBuilder:
    """Build a constrained mesh with reproducible Triangle options (``a`` = maximum triangle area,
    in units of the base-tip length squared)."""

    boundary_tolerance: float = 0.0005
    epsilon: float = 1e-12
    options: str = "pq20a0.001"

    def __post_init__(self) -> None:
        if self.boundary_tolerance <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("meshing tolerances must be positive")

    def build(self, shape: LeafShape2D) -> FlatMesh2D:
        boundary, source_indices = simplify_boundary_inside(
            shape.contour_xy,
            tolerance=self.boundary_tolerance,
            epsilon=self.epsilon,
        )
        indices = np.arange(len(boundary), dtype=np.int32)
        segments = np.column_stack([indices, np.roll(indices, -1)])
        result = triangle_backend.triangulate(
            {"vertices": boundary, "segments": segments}, self.options
        )
        if not {"vertices", "triangles", "segments"} <= set(result):
            raise ValueError("Triangle did not return a complete constrained mesh")
        vertices = np.asarray(result["vertices"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (2,) or not np.all(np.isfinite(vertices)):
            raise ValueError("Triangle returned invalid vertices")
        faces = _canonical_faces(vertices, np.asarray(result["triangles"]), self.epsilon)
        edges = _canonical_edges(np.asarray(result["segments"]))
        input_indices = _map_input_boundary(vertices, boundary, self.epsilon)
        return FlatMesh2D(
            vertices_xy=vertices,
            faces=faces,
            boundary_edges=edges,
            input_boundary_mesh_indices=input_indices,
            boundary_source_indices=source_indices,
        )
