import numpy as np

from leafgen.bend.bend_prior import BendPriorDeformer, BendPriorProvider
from leafgen.contracts.geometry import LeafShape2D
from leafgen.mesh.triangle import TriangleMeshBuilder
from leafgen.paths import WEIGHTS
from leafgen.semantics.mean_value import MeanValueSemanticMapper


def ellipse(n=256):
    t = np.linspace(-np.pi / 2, 3 * np.pi / 2, n, endpoint=False)
    return np.column_stack([0.3 * np.cos(t), 0.5 + 0.5 * np.sin(t)])  # base (0, 0), tip (0, 1)


def test_prior_bend_is_reproducible_and_keeps_the_attachment():
    provider = BendPriorProvider.load(WEIGHTS / "bend_prior.npz", temperature=0.9)
    a = provider.sample(seed=3, role="terminal").parameters
    assert np.array_equal(a, provider.sample(seed=3, role="terminal").parameters)
    shape = LeafShape2D(ellipse(), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    mesh = TriangleMeshBuilder().build(shape)
    semantic = MeanValueSemanticMapper().compute(shape, mesh)
    vertices, attachment, tip, _ = BendPriorDeformer(provider=provider).apply(
        semantic, provider.sample(seed=3, role="terminal")
    )
    assert np.allclose(vertices[:, :2], mesh.vertices_xy)  # the bend only lifts
    assert np.allclose(attachment, 0.0)
    assert np.isfinite(vertices).all() and np.abs(vertices[:, 2]).max() < 0.5
