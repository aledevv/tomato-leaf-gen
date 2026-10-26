import os

import numpy as np
import pytest

from leafgen.leaf import LeafGenerator, write_obj
from leafgen.paths import WEIGHTS

pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "veinflow/model.pt").exists(), reason="weights not downloaded"
)


def test_one_leaf_end_to_end(tmp_path):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    generator = LeafGenerator(WEIGHTS, mesh_area=2e-4, relief_resolution=512)
    leaf = generator.leaf(0)
    assert np.isfinite(leaf.upper).all() and np.isfinite(leaf.lower).all()
    assert len(leaf.tree.flow) > 50  # VeinFlow drew a vein tree
    # the two sides meet on the outline and are apart inside
    assert np.allclose(leaf.upper[leaf.boundary], leaf.lower[leaf.boundary])
    assert (np.linalg.norm(leaf.upper - leaf.lower, axis=1) > 0.1).mean() > 0.5
    obj = write_obj(leaf, tmp_path, "leaf")
    assert obj.read_text().count("\nf ") == 2 * len(leaf.faces)
    again = generator.leaf(0)
    assert np.allclose(leaf.contour, again.contour)  # one seed, one leaflet
