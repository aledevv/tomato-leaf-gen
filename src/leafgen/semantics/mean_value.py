"""Positive mean-value harmonic coordinates on a constrained leaf mesh."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from leafgen.contracts.geometry import (
    FlatMesh2D,
    LeafShape2D,
    SemanticFlatMesh2D,
    signed_area,
)


def _boundary_loop_ccw(mesh: FlatMesh2D) -> np.ndarray:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in np.asarray(mesh.boundary_edges, dtype=np.int64):
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    if not adjacency or any(len(set(values)) != 2 for values in adjacency.values()):
        raise ValueError("flat-mesh boundary must be one degree-2 loop")
    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        choices = [value for value in sorted(set(adjacency[current])) if value != previous]
        if not choices:
            raise ValueError("flat-mesh boundary loop is open")
        following = choices[0]
        if following == start:
            break
        if following in loop:
            raise ValueError("flat mesh has multiple or self-touching boundary loops")
        loop.append(following)
        previous, current = current, following
        if len(loop) > len(adjacency):
            raise ValueError("flat-mesh boundary traversal did not close")
    if len(loop) != len(adjacency):
        raise ValueError("flat mesh has more than one boundary component")
    result = np.asarray(loop, dtype=np.int64)
    if signed_area(mesh.vertices_xy[result]) < 0.0:
        result = np.concatenate([result[:1], result[:0:-1]])
    return result


def _arc(loop: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return loop[start : end + 1]
    return np.concatenate([loop[start:], loop[: end + 1]])


def _boundary_conditions(
    mesh: FlatMesh2D, u: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
    loop = _boundary_loop_ccw(mesh)
    proximal = min(
        (int(index) for index in loop), key=lambda index: (u[index], abs(w[index]), index)
    )
    distal = min(
        (int(index) for index in loop), key=lambda index: (-u[index], abs(w[index]), index)
    )
    if proximal == distal:
        raise ValueError("proximal and distal anchors collapsed")
    positions = {int(index): position for position, index in enumerate(loop)}
    first = _arc(loop, positions[proximal], positions[distal])
    second = _arc(loop, positions[distal], positions[proximal])[::-1]
    first_median = float(np.median(w[first[1:-1]]))
    second_median = float(np.median(w[second[1:-1]]))
    left, right = (first, second) if first_median < second_median else (second, first)
    if not float(np.median(w[left[1:-1]])) < 0.0 < float(np.median(w[right[1:-1]])):
        raise ValueError("boundary arcs do not straddle the biological axis")
    values = np.full(len(mesh.vertices_xy), np.nan, dtype=np.float64)
    values[left[1:-1]] = -1.0
    values[right[1:-1]] = 1.0
    values[[proximal, distal]] = 0.0
    return loop, proximal, distal, left, right, values


def _tan_half_angle(first: np.ndarray, second: np.ndarray) -> float:
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    cross = abs(float(first[0] * second[1] - first[1] * second[0]))
    denominator = first_length * second_length + float(first @ second)
    if cross <= 0.0 or denominator <= 0.0:
        raise ValueError("mean-value Laplacian encountered a degenerate angle")
    return cross / denominator


def _mean_value_laplacian(mesh: FlatMesh2D) -> tuple[csr_matrix, float]:
    weights: dict[tuple[int, int], float] = defaultdict(float)
    vertices = mesh.vertices_xy
    for triangle in np.asarray(mesh.faces, dtype=np.int64):
        for first_pos, second_pos, opposite_pos in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            first = int(triangle[first_pos])
            second = int(triangle[second_pos])
            opposite = int(triangle[opposite_pos])
            edge_length = float(np.linalg.norm(vertices[first] - vertices[second]))
            if edge_length <= 0.0:
                raise ValueError("mean-value Laplacian encountered a zero-length edge")
            contribution = (
                _tan_half_angle(
                    vertices[first] - vertices[opposite],
                    vertices[second] - vertices[opposite],
                )
                / edge_length
            )
            weights[tuple(sorted((first, second)))] += contribution
    if not weights or any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("mean-value weights must be strictly positive")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(len(vertices), dtype=np.float64)
    for (first, second), weight in sorted(weights.items()):
        rows.extend([first, second])
        columns.extend([second, first])
        values.extend([-weight, -weight])
        diagonal[first] += weight
        diagonal[second] += weight
    rows.extend(range(len(vertices)))
    columns.extend(range(len(vertices)))
    values.extend(diagonal.tolist())
    matrix = coo_matrix((values, (rows, columns)), shape=(len(vertices), len(vertices))).tocsr()
    return matrix, min(weights.values())


def _isoline_node(
    first: int,
    second: int,
    values: np.ndarray,
    vertices: np.ndarray,
    u: np.ndarray,
    tolerance: float,
) -> tuple[tuple[str, int, int], np.ndarray, float] | None:
    first_value = float(values[first])
    second_value = float(values[second])
    if abs(first_value) <= tolerance:
        return ("v", first, first), vertices[first], float(u[first])
    if abs(second_value) <= tolerance:
        return ("v", second, second), vertices[second], float(u[second])
    if first_value * second_value >= 0.0:
        return None
    low, high = sorted((first, second))
    fraction = first_value / (first_value - second_value)
    point = vertices[first] + fraction * (vertices[second] - vertices[first])
    point_u = float(u[first] + fraction * (u[second] - u[first]))
    return ("e", low, high), point, point_u


def _extract_centerline(
    mesh: FlatMesh2D,
    values: np.ndarray,
    u: np.ndarray,
    proximal: int,
    distal: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    node_xy: dict[tuple[str, int, int], np.ndarray] = {}
    node_u: dict[tuple[str, int, int], float] = {}
    graph: dict[tuple[str, int, int], set[tuple[str, int, int]]] = defaultdict(set)
    for triangle in np.asarray(mesh.faces, dtype=np.int64):
        nodes: dict[tuple[str, int, int], tuple[np.ndarray, float]] = {}
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            node = _isoline_node(first, second, values, mesh.vertices_xy, u, tolerance)
            if node is not None:
                key, point, point_u = node
                nodes[key] = (point, point_u)
        if len(nodes) <= 1:
            continue
        if len(nodes) != 2:
            raise ValueError("margin_v=0 is ambiguous inside a triangle")
        first_key, second_key = sorted(nodes)
        if first_key == second_key:
            continue
        for key, (point, point_u) in nodes.items():
            node_xy[key] = point
            node_u[key] = point_u
        graph[first_key].add(second_key)
        graph[second_key].add(first_key)
    if not graph:
        raise ValueError("margin_v=0 centerline is empty")
    components = 0
    unseen = set(graph)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbour in graph[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    endpoints = sorted(key for key, neighbours in graph.items() if len(neighbours) == 1)
    expected = [("v", proximal, proximal), ("v", distal, distal)]
    if components != 1 or any(len(values) > 2 for values in graph.values()):
        raise ValueError("margin_v=0 must be one unbranched component")
    if endpoints != sorted(expected):
        raise ValueError("centerline endpoints do not match the boundary anchors")
    ordered = [expected[0]]
    previous: tuple[str, int, int] | None = None
    current = expected[0]
    while current != expected[1]:
        following = sorted(value for value in graph[current] if value != previous)
        if len(following) != 1:
            raise ValueError("centerline ordering is ambiguous")
        previous, current = current, following[0]
        ordered.append(current)
        if len(ordered) > len(graph):
            raise ValueError("centerline contains a cycle")
    centerline_xy = np.vstack([node_xy[key] for key in ordered])
    centerline_u = np.asarray([node_u[key] for key in ordered], dtype=np.float64)
    if np.any(np.diff(centerline_u) <= 1e-12):
        raise ValueError("centerline_u is not strictly increasing")
    return centerline_xy, centerline_u


@dataclass(frozen=True)
class MeanValueSemanticMapper:
    """Compute longitudinal, lateral and harmonic margin coordinates."""

    zero_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.zero_tolerance <= 0.0:
            raise ValueError("zero tolerance must be positive")

    def compute(self, shape: LeafShape2D, mesh: FlatMesh2D) -> SemanticFlatMesh2D:
        frame = shape.attachment_frame
        relative = mesh.vertices_xy - frame.origin_xy
        u = relative @ frame.longitudinal_axis_xy
        w = relative @ frame.lateral_axis_xy
        _, proximal, distal, left, right, boundary_values = _boundary_conditions(mesh, u, w)
        laplacian, _ = _mean_value_laplacian(mesh)
        boundary = np.flatnonzero(np.isfinite(boundary_values))
        interior = np.flatnonzero(~np.isfinite(boundary_values))
        if not len(interior):
            raise ValueError("mean-value field requires interior mesh vertices")
        values = boundary_values.copy()
        system = laplacian[interior][:, interior]
        right_hand = -(laplacian[interior][:, boundary] @ values[boundary])
        values[interior] = spsolve(system, right_hand)
        if not np.all(np.isfinite(values)):
            raise ValueError("mean-value solve returned non-finite values")
        if float(np.max(np.abs(laplacian[interior] @ values))) > 1e-10:
            raise ValueError("mean-value solve exceeded its residual tolerance")
        if float(np.min(values)) < -1.0 - 1e-12 or float(np.max(values)) > 1.0 + 1e-12:
            raise ValueError("mean-value field left the expected range")
        centerline_xy, centerline_u = _extract_centerline(
            mesh, values, u, proximal, distal, self.zero_tolerance
        )
        side = np.zeros(len(mesh.vertices_xy), dtype=np.int8)
        side[left[1:-1]] = -1
        side[right[1:-1]] = 1
        return SemanticFlatMesh2D(
            mesh=mesh,
            longitudinal_u=u,
            lateral_w=w,
            margin_v=values,
            boundary_side=side,
            proximal_boundary_anchor=proximal,
            distal_boundary_anchor=distal,
            geometric_centerline_xy=centerline_xy,
            centerline_u=centerline_u,
        )
