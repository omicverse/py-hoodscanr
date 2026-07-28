"""Soft neighbourhood scanning — the core of hoodscanR.

Mirror of ``hoodscanR/R/soft_neighbourhood.R``.

The idea, in one equation.  Given the distances ``d_ij`` from cell *i* to its
*k* nearest neighbours, the probability that cell *i* "belongs to" neighbour
*j* is a temperature-scaled softmax on **squared** distance::

    P_ij = exp(-d_ij^2 / tau) / sum_j' exp(-d_ij'^2 / tau)

This is what distinguishes hoodscanR from `monkeybread` and `squidpy`, which
both count each neighbour as exactly ``1``.  Here a neighbour 5 um away and a
neighbour 200 um away contribute very differently, and every row of ``P`` is a
genuine probability distribution — which is what makes the entropy in
:mod:`pyhoodscanr.metrics` well defined.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from ._rmath import r_median, r_row_sums, r_sum
from ._roptim import r_optim_bfgs

__all__ = ["scan_hoods", "soft_max_intl", "f_nll"]


def soft_max_intl(m: np.ndarray, t: float, exact: bool = True) -> np.ndarray:
    """``hoodscanR:::soft_max_intl`` — row-wise softmax on ``-m**2 / t``.

    Parameters
    ----------
    m
        Distance matrix, ``n_cells x k``.
    t
        Bandwidth ``tau``.  Larger ``tau`` = flatter (more uniform) weights.
    exact
        Use R's long-double row accumulator (see :mod:`pyhoodscanr._rmath`).
        Set ``False`` for a ~2x faster float64 path; the two agree to ~1 ulp.

    Notes
    -----
    R computes ``exp(-m^2/t)`` **without** subtracting a row maximum.  We do
    the same rather than using the numerically-stabler shifted form, because
    the shift changes the result in the last bits and the parity gate is
    element-wise at 1e-8.  ``d^2/tau`` is O(1) by construction of
    ``tau = median(d^2)/5``, so there is no overflow to protect against on
    real inputs.
    """
    m = np.asarray(m, dtype=np.float64)
    mm = m**2
    mm = -mm / t
    exp_m = np.exp(mm)
    rs = r_row_sums(exp_m) if exact else exp_m.sum(axis=1)
    return exp_m / rs[:, None]


def f_nll(m: np.ndarray, t: float) -> float:
    """``hoodscanR:::f_nll`` — negative log-likelihood of the soft assignment.

    ``-sum(log(P + 1e-8))``.  The ``1e-8`` floor is R's, and it is what stops
    the objective diverging as ``tau -> 0``.
    """
    probm = soft_max_intl(m, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -r_sum(np.log(probm + 1e-8))


def scan_hoods(
    m: np.ndarray,
    mode: Literal["proximityFocused", "smoothFadeout"] = "proximityFocused",
    tau: float | None = None,
    t_init: float | None = None,
    verbose: bool = True,
    return_tau: bool = False,
):
    """``hoodscanR::scanHoods`` — scan cellular neighbourhoods.

    Parameters
    ----------
    m
        Distance matrix (``n_cells x k``), e.g. ``find_near_cells(...)["distance"]``.
    mode
        ``"proximityFocused"`` (default) fixes ``tau = median(m**2) / 5``.
        ``"smoothFadeout"`` fits ``tau`` by maximum likelihood with R's BFGS.
    tau
        Override the bandwidth (``proximityFocused`` only).  R default: ``NA``.
    t_init
        Starting value for the ``smoothFadeout`` optimisation.
        R default: ``median(m**2)``.
    verbose
        Emit R's ``"Tau is set to: ..."`` messages.
    return_tau
        If ``True`` return ``(probability_matrix, tau)`` instead of just the
        matrix.  Python-only convenience; R hides the fitted tau.

    Returns
    -------
    numpy.ndarray
        ``n_cells x k`` probability matrix; every row sums to 1.

    Examples
    --------
    >>> import numpy as np, pyhoodscanr as ph
    >>> rng = np.random.default_rng(0)
    >>> m = np.abs(rng.normal(size=(1000, 100)))
    >>> pm = ph.scan_hoods(m, verbose=False)
    >>> bool(np.allclose(pm.sum(axis=1), 1.0))
    True
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("The input m must be a matrix.")
    if mode not in ("proximityFocused", "smoothFadeout"):
        raise ValueError("mode must be either proximityFocused or smoothFadeout.")

    if mode == "proximityFocused":
        if tau is None:
            tau = r_median(m**2) / 5
    else:  # smoothFadeout
        if t_init is None:
            t_init = r_median(m**2)
        # R: stats::optim(par = t_init, fn = f_nll, m = m, method = "BFGS")
        # `m` is matched by name, so `par` lands on the second formal, `t`.
        opt = r_optim_bfgs(np.array([t_init], dtype=np.float64),
                           lambda p: f_nll(m, p[0]))
        tau = float(opt.par[0])
        if verbose:
            warnings.warn(f"Optimized tau is: {tau}", stacklevel=2)

    if verbose:
        warnings.warn(f"Tau is set to: {tau}", stacklevel=2)

    probm = soft_max_intl(m, tau)
    if return_tau:
        return probm, float(tau)
    return probm
