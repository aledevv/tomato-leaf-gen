"""Protocols and result objects for the modular leaf pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from leafgen.contracts.geometry import (
    FlatMesh2D,
    LeafRole,
    LeafShape2D,
    SemanticFlatMesh2D,
)


@dataclass(frozen=True)
class DeformationSample:
    """One coupled, empirically fitted deformation parameter vector."""

    parameters: np.ndarray
    requested_role: LeafRole
    source_role: str
    source_id: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64)
        if parameters.ndim != 1 or not len(parameters) or not np.all(np.isfinite(parameters)):
            raise ValueError("deformation parameters must be a finite vector")
        if self.requested_role not in {"left", "right", "terminal"}:
            raise ValueError("invalid requested deformation role")
        expected = "terminal" if self.requested_role == "terminal" else "lateral"
        if self.source_role != expected:
            raise ValueError("deformation source role does not match the requested role")
        if not self.source_id:
            raise ValueError("deformation source ID is required")
        result = np.array(parameters, copy=True)
        result.setflags(write=False)
        object.__setattr__(self, "parameters", result)
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class GeneratedLeafAsset:
    """Final V1 asset: one topology with flat and calibrated static poses."""

    instance_id: str
    seed: int
    role: LeafRole
    shape_seed: int
    deformation_seed: int
    shape_attempt: int
    shape: LeafShape2D
    semantic_mesh: SemanticFlatMesh2D
    flat_vertices_xyz: np.ndarray
    calibrated_vertices_xyz: np.ndarray
    flat_attachment_xyz: np.ndarray
    flat_tip_xyz: np.ndarray
    calibrated_attachment_xyz: np.ndarray
    calibrated_tip_xyz: np.ndarray
    attachment_v: float
    deformation_sample: DeformationSample
    rejected_attempt_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.semantic_mesh.mesh.vertices_xy)
        for name in (
            "flat_vertices_xyz",
            "calibrated_vertices_xyz",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite Nx3 array")
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        for name in (
            "flat_attachment_xyz",
            "flat_tip_xyz",
            "calibrated_attachment_xyz",
            "calibrated_tip_xyz",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        expected_flat = np.column_stack(
            [self.semantic_mesh.mesh.vertices_xy, np.zeros(count, dtype=np.float64)]
        )
        if not np.array_equal(self.flat_vertices_xyz, expected_flat):
            raise ValueError("flat pose changed the semantic rest mesh")
        if len(self.rejected_attempt_reasons) != self.shape_attempt - 1:
            raise ValueError("retry history does not match the accepted attempt")

    def vertices_for_pose(self, pose: str) -> np.ndarray:
        if pose == "flat":
            return self.flat_vertices_xyz
        if pose == "calibrated":
            return self.calibrated_vertices_xyz
        raise KeyError(f"unknown pose: {pose}")


class ShapeGenerator(Protocol):
    def generate(self, *, seed: int, role: LeafRole) -> LeafShape2D: ...


class MeshBuilder(Protocol):
    def build(self, shape: LeafShape2D) -> FlatMesh2D: ...


class SemanticMapper(Protocol):
    def compute(self, shape: LeafShape2D, mesh: FlatMesh2D) -> SemanticFlatMesh2D: ...


class DeformationProvider(Protocol):
    def sample(self, *, seed: int, role: LeafRole) -> DeformationSample: ...


class StaticDeformer(Protocol):
    def apply(
        self, mesh: SemanticFlatMesh2D, sample: DeformationSample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...


class LeafPipelineProtocol(Protocol):
    def generate(self, *, seed: int, role: LeafRole) -> GeneratedLeafAsset: ...
