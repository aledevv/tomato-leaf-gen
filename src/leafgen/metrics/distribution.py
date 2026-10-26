"""Set metrics: are generated leaves realistic and as varied as real ones?

Every set is a matrix of rows (one leaf each, in a feature space chosen by the caller) compared
with Euclidean distances.

- Density and Coverage (Naeem et al., ICML 2020), k = 5: each real leaf gets a ball whose radius
  is the distance to its k-th nearest real neighbour. Density = mean number of real balls that
  contain a generated leaf, divided by k (~1 when the distributions match; fidelity).
  Coverage = share of real leaves whose ball contains at least one generated leaf
  (~1 - 2^-k for matching distributions; diversity).
- 1-NNA (Lopez-Paz & Oquab, ICLR 2017; Yang et al., ICCV 2019): leave-one-out 1-nearest-
  neighbour accuracy of real vs generated in the union; 0.5 = indistinguishable.
- Uncertainty: subsampling (80% of each set without replacement, Politis & Romano, Ann. Stat.
  1994), since resampling with replacement creates duplicates with zero distance and breaks the
  nearest-neighbour radii. `interval` gives the raw quantiles of the subsample metrics, which are
  too narrow (an 80% subsample is mostly the same data); `rescaled_interval` is the Politis &
  Romano interval, the subsample deviations rescaled to the full sample size. Reference: the same
  metrics between two random halves of the real set, i.e. what a perfect generator would score at
  this sample size.
- Data copying (Meehan, Chaudhuri & Dasgupta, AISTATS 2020): `z_u` and `data_copying`.
"""

import numpy as np
from scipy.spatial.distance import cdist

K = 5


def density_coverage(rr, rg, k=K):
    """rr: real x real (diagonal inf), rg: real x generated."""
    radius = np.sort(rr, axis=1)[:, k - 1]
    inside = rg <= radius[:, None]
    return dict(
        density=float(inside.sum() / (k * rg.shape[1])), coverage=float(inside.any(1).mean())
    )


def one_nna(rr, gg, rg):
    n, m = rg.shape
    d = np.block([[rr, rg], [rg.T, gg]])
    np.fill_diagonal(d, np.inf)
    labels = np.r_[np.zeros(n), np.ones(m)]
    nearest = np.isclose(d, d.min(1)[:, None], rtol=1e-12, atol=1e-14)
    credit = (nearest & (labels[:, None] == labels[None, :])).sum(1) / nearest.sum(1)
    return float(credit.mean())


def metrics(rr, gg, rg, k=K):
    return dict(**density_coverage(rr, rg, k), one_nna=one_nna(rr, gg, rg))


def subsamples(rr, gg, rg, k=K, fraction=0.8, repeats=200, seed=0):
    """Metrics of `repeats` subsamples (`fraction` of each set, without replacement)."""
    rng = np.random.default_rng(seed)
    n, m = rg.shape
    a, b = max(k + 1, int(fraction * n)), max(k + 1, int(fraction * m))
    runs = []
    for _ in range(repeats):
        i, j = rng.choice(n, a, replace=False), rng.choice(m, b, replace=False)
        runs.append(metrics(rr[np.ix_(i, i)], gg[np.ix_(j, j)], rg[np.ix_(i, j)], k))
    return runs


def interval(rr, gg, rg, k=K, fraction=0.8, repeats=200, seed=0, runs=None):
    runs = runs or subsamples(rr, gg, rg, k, fraction, repeats, seed)
    return {key: np.quantile([r[key] for r in runs], [0.025, 0.975]).tolist() for key in runs[0]}


def rescaled_interval(rr, gg, rg, k=K, fraction=0.8, repeats=200, seed=0, runs=None):
    """95% subsampling interval of Politis & Romano: the quantiles q of (subsample metric - full
    metric) give [full - s * q_97.5, full - s * q_2.5], with s = sqrt(b / (n - b)) since a subsample
    of b out of n drawn without replacement varies as 1/b - 1/n instead of 1/n (s = 2 at 80%)."""
    full = metrics(rr, gg, rg, k)
    runs = runs or subsamples(rr, gg, rg, k, fraction, repeats, seed)
    s = np.sqrt(fraction / (1 - fraction))
    out = {}
    for key, value in full.items():
        lo, hi = np.quantile([r[key] - value for r in runs], [0.025, 0.975])
        out[key] = [float(value - s * hi), float(value - s * lo)]
    return out


def group_halves(rng, groups):
    """Two equal-size halves of the indices with every group (e.g. one leaf scanned at several
    times) entirely on one side, so that no half meets a copy of its own leaves."""
    groups = np.asarray(groups)
    names = rng.permutation(np.unique(groups))
    first = np.isin(groups, names[: len(names) // 2])
    a, b = np.nonzero(first)[0], np.nonzero(~first)[0]
    n = min(len(a), len(b))
    return rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)


def real_vs_real(rr, k=K, repeats=200, seed=0, groups=None):
    """Metrics of one random half of the real set against the other half (halves split by
    `groups` when given)."""
    rng = np.random.default_rng(seed)
    n = len(rr)
    runs = []
    for _ in range(repeats):
        if groups is not None:
            a, b = group_halves(rng, groups)
            runs.append(metrics(rr[np.ix_(a, a)], rr[np.ix_(b, b)], rr[np.ix_(a, b)], k))
            continue
        order = rng.permutation(n)
        a, b = order[: n // 2], order[n // 2 : 2 * (n // 2)]
        runs.append(metrics(rr[np.ix_(a, a)], rr[np.ix_(b, b)], rr[np.ix_(a, b)], k))
    return {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}


# ----------------------------------------------------------------------------- data copying
def distances_to(points, train, chunk=1024):
    """Euclidean distance of every row of ``points`` to its two nearest rows of ``train``."""
    out = np.empty((len(points), 2))
    for start in range(0, len(points), chunk):
        d = cdist(points[start : start + chunk], train)
        out[start : start + chunk] = np.sort(d, axis=1)[:, :2]
    return out


def z_u(held_distances, generated_distances):
    """Meehan, Chaudhuri & Dasgupta (AISTATS 2020), eq. of section 3.1: U = pairs with
    d(generated) > d(held-out), both distances to the training set; z-scored under H0.
    Z_U << 0 = data copying, Z_U >> 0 = underfitting."""
    from scipy.stats import mannwhitneyu

    n, m = len(held_distances), len(generated_distances)
    u = float(mannwhitneyu(generated_distances, held_distances, alternative="less").statistic)
    return (u - n * m / 2) / np.sqrt(n * m * (n + m + 1) / 12)


def data_copying(train, held, generated, cells=10, tau=None, seed=0):
    """Global Z_U, the cell-averaged C_T (cells = k-means on the training set, only cells holding
    at least a share ``tau`` of the generated samples, weighted by the held-out share; Meehan et
    al. 2020, section 3.2), and two companions: the median distance to the closest training
    sample (DCR) of generated and held-out, and the share of generated samples whose
    nearest/second-nearest training distance ratio is below 1/3 (Yoon et al. 2023)."""
    from sklearn.cluster import KMeans

    train, held, generated = (np.asarray(x, dtype=np.float64) for x in (train, held, generated))
    tau = 20 / len(generated) if tau is None else tau
    dh, dg = distances_to(held, train), distances_to(generated, train)
    km = KMeans(cells, n_init=3, random_state=seed).fit(train)
    ch, cg = km.predict(held), km.predict(generated)
    scores, weights = [], []
    for cell in range(cells):
        in_h, in_g = ch == cell, cg == cell
        if in_g.sum() / len(generated) < tau or in_h.sum() < 2 or in_g.sum() < 2:
            continue
        scores.append(z_u(dh[in_h, 0], dg[in_g, 0]))
        weights.append(in_h.sum() / len(held))
    return dict(
        z_u=float(z_u(dh[:, 0], dg[:, 0])),
        c_t=float(np.dot(weights, scores) / np.sum(weights)) if weights else float("nan"),
        cells_used=len(scores),
        cells=cells,
        dcr_median_generated=float(np.median(dg[:, 0])),
        dcr_median_held=float(np.median(dh[:, 0])),
        nn_ratio_below_third=float(np.mean(dg[:, 0] / np.maximum(dg[:, 1], 1e-12) < 1 / 3)),
    )
