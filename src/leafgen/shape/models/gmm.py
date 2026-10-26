"""Full-covariance Gaussian mixture on latent codes, size chosen by BIC."""

from __future__ import annotations

import numpy as np

MAX_COMPONENTS = 12


def _fit_gmm(scores: np.ndarray, components: int, seed: int) -> tuple[dict, dict]:
    from sklearn.mixture import GaussianMixture

    candidates = [components] if components > 0 else list(range(1, MAX_COMPONENTS + 1))
    best, bic = None, {}
    for k in candidates:
        g = GaussianMixture(
            k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-6
        ).fit(scores)
        bic[k] = float(g.bic(scores))
        if best is None or bic[k] < bic[best.n_components]:
            best = g
    arrays = dict(weights=best.weights_, means=best.means_, covariances=best.covariances_)
    return arrays, dict(kind="gmm", components=int(best.n_components), bic=bic)
