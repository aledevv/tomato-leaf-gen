import numpy as np
from scipy.spatial.distance import cdist

from leafgen.metrics.distribution import metrics, z_u


def _sets(a, b):
    rr, gg, rg = cdist(a, a), cdist(b, b), cdist(a, b)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)
    return rr, gg, rg


def test_same_distribution_is_indistinguishable():
    rng = np.random.default_rng(0)
    m = metrics(*_sets(rng.normal(size=(400, 4)), rng.normal(size=(400, 4))))
    assert abs(m["one_nna"] - 0.5) < 0.06
    assert 0.8 < m["density"] < 1.2
    assert m["coverage"] > 0.9


def test_shifted_distribution_is_detected():
    rng = np.random.default_rng(0)
    m = metrics(*_sets(rng.normal(size=(400, 4)), rng.normal(3.0, size=(400, 4))))
    assert m["one_nna"] > 0.95
    assert m["coverage"] < 0.2


def test_copies_give_negative_z_u():
    rng = np.random.default_rng(0)
    held = rng.uniform(0.5, 1.0, 300)  # distances of real held-out leaves to the training set
    copies = rng.uniform(0.0, 0.1, 300)  # draws sitting on training leaves
    assert z_u(held, copies) < -10
