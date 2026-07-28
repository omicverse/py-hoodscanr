"""Bit-faithful port of R's ``stats::optim(method = "BFGS")``.

Why this file exists
--------------------
``scanHoods(mode = "smoothFadeout")`` fits the softmax bandwidth tau by

    stats::optim(par = t_init, fn = f_nll, m = m, method = "BFGS")

``scipy.optimize.minimize(method="BFGS")`` is **not** the same algorithm.  Two
differences change the answer materially here:

1. **Finite-difference step.**  R's ``optim`` uses an *absolute* central
   difference with ``ndeps = 1e-3`` (``control$ndeps``).  SciPy uses a
   *relative* forward/central step of order ``sqrt(eps)``.  Since tau is
   O(1e5) on real data, R's step is a ~1e-8 relative perturbation and SciPy's
   is ~1e-8 absolute — completely different gradients.
2. **Line search.**  R's ``vmmin`` uses a backtracking search with
   ``stepredn = 0.2`` and Armijo constant ``acctol = 1e-4``; SciPy uses a
   strong-Wolfe cubic/quadratic interpolation search.

Porting ``vmmin`` verbatim is the only way to land on the same tau.

Reference: R 4.4.3 ``src/appl/optim.c`` (``vmmin``) and
``src/library/stats/src/optim.c`` (``fmingr`` numeric-gradient branch,
``do_optim`` defaults).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["r_optim_bfgs", "ROptimResult"]

# optim.c
_STEPREDN = 0.2
_ACCTOL = 1.0e-4
_RELTEST = 10.0

# do_optim() defaults for method = "BFGS"
_DEFAULT_MAXIT = 100
_DEFAULT_RELTOL = np.sqrt(np.finfo(np.float64).eps)  # 1.4901161193847656e-08
_DEFAULT_ABSTOL = -np.inf
_DEFAULT_NDEPS = 1.0e-3


class ROptimResult:
    """Mirror of R's ``optim`` return value."""

    __slots__ = ("par", "value", "counts", "convergence")

    def __init__(self, par, value, counts, convergence):
        self.par = par
        self.value = value
        self.counts = counts
        self.convergence = convergence

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"ROptimResult(par={self.par!r}, value={self.value!r}, "
            f"counts={self.counts!r}, convergence={self.convergence})"
        )


def _numeric_gradient(
    fn: Callable[[np.ndarray], float],
    par: np.ndarray,
    ndeps: float,
) -> np.ndarray:
    """R ``stats/src/optim.c::fmingr`` numeric branch, ``parscale = fnscale = 1``.

    ``df[i] = (f(p + eps) - f(p - eps)) / (2 * eps)`` with an **absolute**
    ``eps = ndeps[i]``.
    """
    n = par.shape[0]
    df = np.empty(n, dtype=np.float64)
    x = par.copy()
    for i in range(n):
        eps = ndeps
        x[i] = par[i] + eps
        val1 = fn(x)
        x[i] = par[i] - eps
        val2 = fn(x)
        if not np.isfinite(val1) or not np.isfinite(val2):
            raise FloatingPointError(
                "non-finite finite-difference value; R would abort here too"
            )
        df[i] = (val1 - val2) / (2 * eps)
        x[i] = par[i]
    return df


def r_optim_bfgs(
    par,
    fn: Callable[[np.ndarray], float],
    gr: Callable[[np.ndarray], np.ndarray] | None = None,
    maxit: int = _DEFAULT_MAXIT,
    reltol: float = _DEFAULT_RELTOL,
    abstol: float = _DEFAULT_ABSTOL,
    ndeps: float = _DEFAULT_NDEPS,
) -> ROptimResult:
    """``stats::optim(par, fn, gr, method = "BFGS")``.

    Verbatim port of ``vmmin`` from R's ``src/appl/optim.c``.  Variable names
    are kept identical to the C source so the two can be diffed by eye.

    Parameters
    ----------
    par
        Starting value(s).
    fn
        Objective, called with a 1-D float array.
    gr
        Analytic gradient.  If ``None`` (R's default), the same absolute
        central-difference gradient R uses is applied.
    maxit, reltol, abstol, ndeps
        R ``control`` defaults for BFGS.
    """
    b = np.atleast_1d(np.asarray(par, dtype=np.float64)).copy()
    n0 = b.shape[0]

    if gr is None:

        def _fmingr(p):
            return _numeric_gradient(fn, p, ndeps)

    else:

        def _fmingr(p):
            return np.atleast_1d(np.asarray(gr(p), dtype=np.float64))

    if maxit <= 0:
        return ROptimResult(b.copy(), float(fn(b)), {"function": 0, "gradient": 0}, 0)

    # mask[] is all-TRUE in do_optim (no fixed parameters), so l = 0..n0-1
    l = np.arange(n0)
    n = n0

    g = np.zeros(n0, dtype=np.float64)
    t = np.zeros(n, dtype=np.float64)
    X = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    B = np.zeros((n, n), dtype=np.float64)  # lower triangle used

    f = float(fn(b))
    if not np.isfinite(f):
        raise FloatingPointError("initial value in 'vmmin' is not finite")
    Fmin = f
    funcount = gradcount = 1
    g[:] = _fmingr(b)
    iter_ = 1
    ilast = gradcount
    count = 0

    while True:
        if ilast == gradcount:
            B[:] = 0.0
            for i in range(n):
                B[i, i] = 1.0

        for i in range(n):
            X[i] = b[l[i]]
            c[i] = g[l[i]]

        gradproj = 0.0
        for i in range(n):
            s = 0.0
            for j in range(0, i + 1):
                s -= B[i, j] * g[l[j]]
            for j in range(i + 1, n):
                s -= B[j, i] * g[l[j]]
            t[i] = s
            gradproj += s * g[l[i]]

        if gradproj < 0.0:  # search direction is downhill
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[l[i]] = X[i] + steplength * t[i]
                    if _RELTEST + X[i] == _RELTEST + b[l[i]]:  # no change
                        count += 1
                if count < n:
                    f = float(fn(b))
                    funcount += 1
                    accpoint = np.isfinite(f) and (
                        f <= Fmin + gradproj * steplength * _ACCTOL
                    )
                    if not accpoint:
                        steplength *= _STEPREDN
                if count == n or accpoint:
                    break

            enough = (f > abstol) and abs(f - Fmin) > reltol * (abs(Fmin) + reltol)
            # stop if value is small or if relative change is low
            if not enough:
                count = n
                Fmin = f
            if count < n:  # making progress
                Fmin = f
                g[:] = _fmingr(b)
                gradcount += 1
                iter_ += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[l[i]] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(0, i + 1):
                            s += B[i, j] * c[j]
                        for j in range(i + 1, n):
                            s += B[j, i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(0, i + 1):
                            B[i, j] += (
                                D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]
                            ) / D1
                else:  # D1 <= 0
                    ilast = gradcount
            else:  # no progress
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:  # uphill search
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
            # resets unless it has just been reset

        if iter_ >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount  # periodic restart

        if count == n and ilast == gradcount:
            break

    fail = 0 if iter_ < maxit else 1
    return ROptimResult(
        b.copy(), Fmin, {"function": funcount, "gradient": gradcount}, fail
    )
