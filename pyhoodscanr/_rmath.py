"""R-faithful reductions.

R accumulates ``sum()``, ``rowSums()`` and ``mean()`` over doubles in an
80-bit ``LDOUBLE`` register (``src/main/summary.c::rsum``,
``src/main/array.c::do_colsum``).  NumPy accumulates in ``float64`` with
pairwise blocking.

For the softmax itself the difference is ~1 ulp and irrelevant.  It is *not*
irrelevant inside the ``smoothFadeout`` objective: R's ``optim`` probes the
negative log-likelihood with an absolute step of ``1e-3`` on a parameter of
order ``1e5``, so the finite-difference gradient is a cancellation of two
numbers agreeing to ~9 significant figures.  A 1-ulp difference in the
objective is amplified by ~1e12 in the gradient.  Matching R's accumulator is
what makes the fitted tau reproduce.

On x86-64 Linux ``np.longdouble`` is the same 80-bit extended type R uses.
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_sum", "r_row_sums", "r_median", "r_mean", "r_cor", "HAS_LONG_DOUBLE"]

#: True when ``np.longdouble`` really is 80-bit extended (x86-64), i.e. when
#: the accumulator matches R's ``LDOUBLE``.  On platforms where it is merely
#: an alias for ``float64`` (e.g. aarch64 macOS) the reductions silently fall
#: back to ``float64`` and ``tau`` from ``smoothFadeout`` may differ in the
#: last few digits.
HAS_LONG_DOUBLE = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


def r_sum(x: np.ndarray) -> float:
    """``base::sum`` — long-double accumulator, double result."""
    return float(np.asarray(x, dtype=np.longdouble).sum())


def r_row_sums(x: np.ndarray) -> np.ndarray:
    """``base::rowSums`` — long-double per-row accumulator, double result."""
    return np.asarray(x, dtype=np.longdouble).sum(axis=1).astype(np.float64)


def r_median(x: np.ndarray) -> float:
    """``stats::median`` on a numeric matrix/vector (all elements, no NA)."""
    return float(np.median(np.asarray(x, dtype=np.float64)))


def r_mean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``base::mean`` — R's two-pass refined mean (``src/main/summary.c``).

    ``m1 = sum(x)/n``; ``m = m1 + sum(x - m1)/n``.  The second pass recovers
    the bits lost in the first.
    """
    x = np.asarray(x, dtype=np.float64)
    xl = x.astype(np.longdouble)
    n = x.shape[axis]
    m1 = (xl.sum(axis=axis) / n).astype(np.float64)
    shape = list(x.shape)
    shape[axis] = 1
    corr = ((xl - m1.reshape(shape).astype(np.longdouble)).sum(axis=axis) / n)
    return (m1 + corr.astype(np.float64)).astype(np.float64)


def r_cor(x: np.ndarray) -> np.ndarray:
    """``stats::cor`` on the columns of ``x`` (Pearson, complete data).

    Verbatim strategy from R ``src/library/stats/src/cov.c``: refined
    column means, **then** centred cross-products accumulated in long double,
    divided by ``n - 1``, then normalised by the square roots of the diagonal
    and clamped to 1.

    This matters.  ``pandas.DataFrame.corr`` uses the one-pass
    ``E[xy] - E[x]E[y]`` formula, which catastrophically cancels when the
    column means are large relative to the variance — on hoodscanR's
    probability matrix that costs ~4e-8 of accuracy, enough to fail an
    element-wise 1e-8 gate against R.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    xm = r_mean(x, axis=0)
    xc = (x.astype(np.longdouble) - xm.astype(np.longdouble))
    cov = (xc.T @ xc / (n - 1)).astype(np.float64)
    sd = np.sqrt(np.diag(cov))
    out = cov / np.outer(sd, sd)
    return np.clip(out, None, 1.0)
