"""Provider-independent geometry contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

LeafRole = Literal["left", "right", "terminal"]


def signed_area(points: np.ndarray) -> float:
    """Return the signed area of an open polygon loop."""
    contour = np.asarray(points, dtype=np.float64)
    return float(
        0.5
        * np.sum(
            contour[:, 0] * np.roll(contour[:, 1], -1) - np.roll(contour[:, 0], -1) * contour[:, 1]
        )
    )


def contour_self_intersects(points: np.ndarray, *, epsilon: float = 1e-12) -> bool:
    """Return whether non-adjacent polygon edges intersect."""
    contour = np.asarray(points, dtype=np.float64)
    count = len(contour)
    for first in range(count):
        a = contour[first]
        b = contour[(first + 1) % count]
        second = np.arange(first + 2, count)
        if first == 0:
            second = second[second != count - 1]
        if not len(second):
            continue
        c = contour[second]
        d = contour[(second + 1) % count]
        ab = b - a
        cd = d - c
        o1 = ab[0] * (c[:, 1] - a[1]) - ab[1] * (c[:, 0] - a[0])
        o2 = ab[0] * (d[:, 1] - a[1]) - ab[1] * (d[:, 0] - a[0])
        o3 = cd[:, 0] * (a[1] - c[:, 1]) - cd[:, 1] * (a[0] - c[:, 0])
        o4 = cd[:, 0] * (b[1] - c[:, 1]) - cd[:, 1] * (b[0] - c[:, 0])
        if np.any((o1 * o2 < -(epsilon**2)) & (o3 * o4 < -(epsilon**2))):
            return True
    return False


def _readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LeafAttachmentFrame:
    """Right-handed biological frame from attachment to tip."""

    origin_xy: np.ndarray
    longitudinal_axis_xy: np.ndarray
    lateral_axis_xy: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xy, dtype=np.float64)
        longitudinal = np.asarray(self.longitudinal_axis_xy, dtype=np.float64)
        lateral = np.asarray(self.lateral_axis_xy, dtype=np.float64)
        if any(value.shape != (2,) for value in (origin, longitudinal, lateral)):
            raise ValueError("attachment-frame vectors must contain two values")
        if not np.all(np.isfinite(np.vstack([origin, longitudinal, lateral]))):
            raise ValueError("attachment-frame vectors must be finite")
        if not np.isclose(np.linalg.norm(longitudinal), 1.0, atol=1e-10):
            raise ValueError("longitudinal axis must have unit length")
        if not np.isclose(np.linalg.norm(lateral), 1.0, atol=1e-10):
            raise ValueError("lateral axis must have unit length")
        if not np.isclose(float(longitudinal @ lateral), 0.0, atol=1e-10):
            raise ValueError("attachment-frame axes must be perpendicular")
        if float(np.linalg.det(np.column_stack([lateral, longitudinal]))) <= 0.0:
            raise ValueError("attachment frame must preserve chirality")
        object.__setattr__(self, "origin_xy", _readonly(origin, np.float64))
        object.__setattr__(self, "longitudinal_axis_xy", _readonly(longitudinal, np.float64))
        object.__setattr__(self, "lateral_axis_xy", _readonly(lateral, np.float64))


@dataclass(frozen=True)
class LeafShape2D:
    """Validated 2D outline with explicit biological landmarks."""

    contour_xy: np.ndarray
    biological_base_xy: np.ndarray
    biological_tip_xy: np.ndarray
    coordinate_frame: str = "base=(0,0), tip=(0,1), right-handed, +Y tip"
    units: Literal["normalized", "metres"] = "normalized"
    provenance: Mapping[str, Any] | None = None
    features_h60: np.ndarray | None = None

    def with_provenance(self, provenance: Mapping[str, Any]) -> LeafShape2D:
        """Replace metadata on validated immutable geometry without revalidating its polygon."""
        metadata = dict(provenance)
        if metadata.get("reflection_applied") is True:
            raise ValueError("reflected shapes are not accepted")
        result = copy(self)
        object.__setattr__(result, "provenance", MappingProxyType(metadata))
        return result

    def __post_init__(self) -> None:
        contour = np.asarray(self.contour_xy, dtype=np.float64)
        base = np.asarray(self.biological_base_xy, dtype=np.float64)
        tip = np.asarray(self.biological_tip_xy, dtype=np.float64)
        if contour.ndim != 2 or contour.shape[1:] != (2,) or len(contour) < 16:
            raise ValueError("contour must be an Nx2 array with at least 16 points")
        if base.shape != (2,) or tip.shape != (2,):
            raise ValueError("base and tip must be 2-vectors")
        if not np.all(np.isfinite(contour)) or not np.all(np.isfinite([base, tip])):
            raise ValueError("shape geometry must be finite")
        if np.linalg.norm(tip - base) <= 1e-12:
            raise ValueError("base and tip must be distinct")
        if signed_area(contour) <= 1e-12:
            raise ValueError("contour must be non-degenerate and counter-clockwise")
        if contour_self_intersects(contour):
            raise ValueError("contour must not self-intersect")
        if self.units not in {"normalized", "metres"}:
            raise ValueError("units must be normalized or metres")
        provenance = dict(self.provenance or {})
        if provenance.get("reflection_applied") is True:
            raise ValueError("reflected shapes are not accepted")
        object.__setattr__(self, "contour_xy", _readonly(contour, np.float64))
        object.__setattr__(self, "biological_base_xy", _readonly(base, np.float64))
        object.__setattr__(self, "biological_tip_xy", _readonly(tip, np.float64))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        if self.features_h60 is not None:
            features = np.asarray(self.features_h60, dtype=np.float64)
            if features.shape != (242,) or not np.all(np.isfinite(features)):
                raise ValueError("features_h60 must be a finite 242-vector")
            object.__setattr__(self, "features_h60", _readonly(features, np.float64))

    @property
    def attachment_frame(self) -> LeafAttachmentFrame:
        direction = self.biological_tip_xy - self.biological_base_xy
        longitudinal = direction / np.linalg.norm(direction)
        lateral = np.asarray([longitudinal[1], -longitudinal[0]], dtype=np.float64)
        return LeafAttachmentFrame(self.biological_base_xy, longitudinal, lateral)


@dataclass(frozen=True)
class FlatMesh2D:
    """Triangular mesh with its constrained boundary mappings."""

    vertices_xy: np.ndarray
    faces: np.ndarray
    boundary_edges: np.ndarray
    input_boundary_mesh_indices: np.ndarray
    boundary_source_indices: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_xy, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.uint32)
        edges = np.asarray(self.boundary_edges, dtype=np.uint32)
        mesh_indices = np.asarray(self.input_boundary_mesh_indices, dtype=np.uint32)
        source_indices = np.asarray(self.boundary_source_indices, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3:
            raise ValueError("mesh vertices must be an Nx2 array")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError("mesh faces must be a non-empty Mx3 array")
        if edges.ndim != 2 or edges.shape[1:] != (2,) or not len(edges):
            raise ValueError("boundary edges must be a non-empty Kx2 array")
        if mesh_indices.ndim != 1 or source_indices.shape != mesh_indices.shape:
            raise ValueError("boundary mappings must be aligned vectors")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices must be finite")
        if np.any(faces >= len(vertices)) or np.any(edges >= len(vertices)):
            raise ValueError("mesh contains invalid indices")
        if np.any(mesh_indices >= len(vertices)) or len(np.unique(mesh_indices)) != len(
            mesh_indices
        ):
            raise ValueError("input boundary mapping is invalid")
        for name, value, dtype in (
            ("vertices_xy", vertices, np.float64),
            ("faces", faces, np.uint32),
            ("boundary_edges", edges, np.uint32),
            ("input_boundary_mesh_indices", mesh_indices, np.uint32),
            ("boundary_source_indices", source_indices, np.uint32),
        ):
            object.__setattr__(self, name, _readonly(value, dtype))


@dataclass(frozen=True)
class SemanticFlatMesh2D:
    """Flat mesh augmented with biological and harmonic control fields."""

    mesh: FlatMesh2D
    longitudinal_u: np.ndarray
    lateral_w: np.ndarray
    margin_v: np.ndarray
    boundary_side: np.ndarray
    proximal_boundary_anchor: int
    distal_boundary_anchor: int
    geometric_centerline_xy: np.ndarray
    centerline_u: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.mesh.vertices_xy)
        arrays = {
            "longitudinal_u": (self.longitudinal_u, np.float64, (count,)),
            "lateral_w": (self.lateral_w, np.float64, (count,)),
            "margin_v": (self.margin_v, np.float64, (count,)),
            "boundary_side": (self.boundary_side, np.int8, (count,)),
            "geometric_centerline_xy": (
                self.geometric_centerline_xy,
                np.float64,
                None,
            ),
            "centerline_u": (self.centerline_u, np.float64, None),
        }
        converted: dict[str, np.ndarray] = {}
        for name, (value, dtype, expected) in arrays.items():
            array = np.asarray(value, dtype=dtype)
            if expected is not None and array.shape != expected:
                raise ValueError(f"{name} does not align with the mesh")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            converted[name] = array
        centerline = converted["geometric_centerline_xy"]
        center_u = converted["centerline_u"]
        if centerline.ndim != 2 or centerline.shape[1:] != (2,) or len(centerline) < 2:
            raise ValueError("centerline must be an Nx2 polyline")
        if center_u.shape != (len(centerline),):
            raise ValueError("centerline_u must align with the centerline")
        if not set(np.unique(converted["boundary_side"])).issubset({-1, 0, 1}):
            raise ValueError("boundary-side codes must be -1, 0 or 1")
        anchors = (self.proximal_boundary_anchor, self.distal_boundary_anchor)
        if any(not isinstance(value, int) or not 0 <= value < count for value in anchors):
            raise ValueError("semantic anchors must be valid mesh indices")
        if anchors[0] == anchors[1]:
            raise ValueError("semantic anchors must be distinct")
        for name, array in converted.items():
            object.__setattr__(self, name, _readonly(array, array.dtype))
