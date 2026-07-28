"""R RNG / R optim / Hartigan-Wong ports, checked against hard-coded R output.

The expected values were produced by R 4.4.3 and are pasted verbatim, so these
tests run without R installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhoodscanr._kmeans_hw import kmeans_hartigan_wong
from pyhoodscanr._roptim import r_optim_bfgs
from pyhoodscanr._rrng import RRandom
from pyhoodscanr._rrng_fast import (
    HAVE_NUMBA,
    perm_le_counts,
    sample_int_kernel,
    state_from,
)


# R: set.seed(42); sprintf("%.17g", runif(5))
R_RUNIF_42 = [
    0.91480604349635541,
    0.93707541329786181,
    0.28613953478634357,
    0.83044762606732547,
    0.64174551889300346,
]

# R: set.seed(42); sample.int(2661, 10)
R_SAMPLE_42 = [2609, 2369, 1177, 1098, 1252, 634, 2097, 1152, 1327, 2072]

# R: set.seed(1); sample.int(10)
R_SAMPLE_1 = [9, 4, 7, 1, 2, 5, 3, 10, 6, 8]


def test_unif_rand_matches_r():
    rng = RRandom(42)
    got = [rng.unif_rand() for _ in range(5)]
    for g, e in zip(got, R_RUNIF_42):
        assert g == e, f"{g!r} != {e!r}"  # bit-exact, not approx


def test_sample_int_matches_r():
    assert list(RRandom(42).sample_int(2661, 10) + 1) == R_SAMPLE_42
    assert list(RRandom(1).sample_int(10) + 1) == R_SAMPLE_1


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("seed", [42, 1, 7, 123456])
def test_compiled_rng_is_bit_identical(seed):
    a = RRandom(seed).sample_int(2661, 10)
    b = sample_int_kernel(state_from(RRandom(seed)), 2661, 10)
    assert np.array_equal(a, b)

    a = RRandom(seed).sample_int(1000)
    b = sample_int_kernel(state_from(RRandom(seed)), 1000, 1000)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_perm_counts_match_interpreted_loop():
    observed = np.random.default_rng(0).random(400)
    rng = RRandom(42)
    counts = np.zeros(400, dtype=np.int64)
    for _ in range(60):
        counts += observed[rng.sample_int(400)] <= observed
    fast = perm_le_counts(state_from(RRandom(42)), observed, 60)
    assert np.array_equal(counts, fast)


def test_r_optim_bfgs_matches_r_on_rosenbrock():
    """Independent check of the vmmin port on a 2-D problem.

    R 4.4.3::

        optim(c(-1.2, 1),
              function(p) (1-p[1])^2 + 100*(p[2]-p[1]^2)^2, method = "BFGS")

        $par    0.99980443323139745  0.99960838062348123
        $value  3.8273827561079511e-08
        $counts function 118  gradient 38

    The call counts are the sharp evidence and they are checked exactly: matching
    R on 118 function and 38 gradient evaluations means the line search, the
    finite-difference gradient and the convergence test all reproduce step for
    step, not merely that both landed near the same minimum.

    The coordinates are checked to 1e-12 rather than with ``==``. An earlier
    version asserted exact float equality and passed on the machine the port was
    written on; on CI runners it failed by 4.5e-14 — the last two digits — because
    BFGS is iterative and different BLAS/libm builds diverge in the final ulps. A
    tolerance of 1e-12 on a coordinate near 1.0 is still agreement to eleven
    significant figures, and unlike ``==`` it is a claim that holds on hardware
    other than the author's.
    """

    def rosen(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    res = r_optim_bfgs(np.array([-1.2, 1.0]), rosen)
    assert res.par[0] == pytest.approx(0.99980443323139745, abs=1e-12)
    assert res.par[1] == pytest.approx(0.99960838062348123, abs=1e-12)
    assert res.value == pytest.approx(3.8273827561079511e-08, rel=1e-9)
    assert res.counts == {"function": 118, "gradient": 38}
    assert res.convergence == 0


def test_hartigan_wong_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(loc, 0.05, (60, 2)) for loc in (0.0, 5.0, 10.0)])
    centers = x[[0, 60, 120]].copy()
    res = kmeans_hartigan_wong(x, centers, iter_max=100)
    lab = res["cluster"]
    assert len(set(lab)) == 3
    for blk in (slice(0, 60), slice(60, 120), slice(120, 180)):
        assert len(set(lab[blk])) == 1
    # within-cluster SS must not exceed the initial assignment's
    assert res["wss"].sum() < ((x - x.mean(0)) ** 2).sum()


def test_hartigan_wong_beats_or_matches_lloyd():
    """Hartigan-Wong is a strictly stronger local search than Lloyd; on the
    same initialisation its total WSS must never be worse."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(7)
    x = rng.random((400, 4))
    centers = x[rng.choice(400, 6, replace=False)].copy()
    hw = kmeans_hartigan_wong(x, centers.copy(), iter_max=1000)
    lloyd = KMeans(n_clusters=6, init=centers.copy(), n_init=1,
                   algorithm="lloyd", max_iter=1000).fit(x)
    assert hw["wss"].sum() <= lloyd.inertia_ + 1e-9
