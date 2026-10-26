"""Contracts shared by the shape, mesh, semantic and bend modules."""

from leafgen.contracts.core import DeformationSample
from leafgen.contracts.geometry import (
    FlatMesh2D,
    LeafAttachmentFrame,
    LeafShape2D,
    SemanticFlatMesh2D,
)

__all__ = [
    "DeformationSample",
    "FlatMesh2D",
    "LeafAttachmentFrame",
    "LeafShape2D",
    "SemanticFlatMesh2D",
]
