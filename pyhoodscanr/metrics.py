"""Per-cell entropy, perplexity and the perplexity permutation test.

Mirror of ``hoodscanR/R/calc_metrics.R`` and ``hoodscanR/src/cal_metrics.cpp``.

This is the part of hoodscanR that has **no** Python equivalent.  `monkeybread`
and `squidpy` both z-score their neighbourhood counts, which destroys the
``sum = 1`` property and leaves entropy undefined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rrng import RRandom

__all__ = ["calculate_metrics", "calc_metrics", "perplexity_permute"]


def calculate_metrics(pm) -> pd.DataFrame:
    """``hoodscanR:::calculate_metrics`` (the Rcpp kernel) — entropy + perplexity.

    ``H(x) = -sum_j P_j log2 P_j`` over the strictly positive entries, and
    ``perplexity = 2^H``.

    Perplexity is the interpretable one: a value of 1 means the cell sits in a
    single, distinct neighbourhood; a value of 2 means it sits on a ~50/50
    boundary between two; and so on — it is the *effective number of
    neighbourhood types* around the cell.

    Returns
    -------
    pandas.DataFrame
        Columns ``entropy`` and ``perplexity``, one row per cell.
    """
    p = np.asarray(pm, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("pm must be a 2-D probability matrix.")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    entropy = -terms.sum(axis=1)
    perplexity = np.power(2.0, entropy)
    return pd.DataFrame({"entropy": entropy, "perplexity": perplexity})


def calc_metrics(adata, pm=None, pm_cols=None, val_names=("entropy", "perplexity")):
    """``hoodscanR::calcMetrics`` — write entropy/perplexity into ``adata.obs``.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData`.
    pm
        Probability matrix.  If ``None``, read it from ``adata.obs[pm_cols]``.
    pm_cols
        Column names of the probability matrix in ``adata.obs``.  If both
        ``pm`` and ``pm_cols`` are ``None``, falls back to
        ``adata.uns['hoodscanr']['pm_cols']``.
    val_names
        Names of the two output columns.
    """
    pm = _resolve_pm(adata, pm, pm_cols)
    res = calculate_metrics(pm)
    adata.obs[val_names[0]] = res["entropy"].to_numpy()
    adata.obs[val_names[1]] = res["perplexity"].to_numpy()
    return adata


def perplexity_permute(adata, pm=None, pm_cols=None, n_perm: int = 1000, seed: int = 42,
                       val_name: str = "perplexity_p", exact: bool = True):
    """``hoodscanR::perplexityPermute`` — permutation p-value for perplexity.

    For each cell, the observed perplexity is compared against the perplexity
    distribution obtained by randomly re-assigning whole neighbourhood
    profiles across cells::

        p_i = (#{perm perplexity <= observed_i} + 1) / (n_perm + 1)

    A **small** p means the cell's neighbourhood is unusually *distinct*
    (low perplexity) relative to the tissue as a whole.

    Parameters
    ----------
    n_perm
        Number of permutations.  R default 1000.
    seed
        Seed passed to the ported R RNG, so the draw matches
        ``set.seed(seed)`` in R exactly.
    exact
        Use the row-permutation-equivariance identity (see ``MATH.md``) to
        avoid recomputing entropy ``n_perm`` times.  Mathematically exact;
        set ``False`` for the literal transcription of the R loop.

    Notes
    -----
    R permutes the **rows of the whole probability matrix**
    (``pm[sample(1:nrow(pm)), ]``).  Since perplexity is computed row-wise,
    the permuted perplexity vector is exactly the observed vector permuted —
    which is what ``exact=True`` exploits.
    """
    pm = _resolve_pm(adata, pm, pm_cols)
    pm_arr = np.asarray(pm, dtype=np.float64)
    observed = calculate_metrics(pm_arr)["perplexity"].to_numpy()
    n = pm_arr.shape[0]

    rng = RRandom(seed)
    if exact:
        from ._rrng_fast import HAVE_NUMBA, perm_le_counts, state_from, state_into

        if HAVE_NUMBA:
            # identical RNG stream and identical comparisons, compiled;
            # ~130x faster than the interpreted loop.  See MATH.md section 2.
            state = state_from(rng)
            counts = perm_le_counts(state, observed, n_perm)
            state_into(rng, state)
        else:
            counts = np.zeros(n, dtype=np.int64)
            for _ in range(n_perm):
                counts += observed[rng.sample_int(n)] <= observed
    else:
        counts = np.zeros(n, dtype=np.int64)
        for _ in range(n_perm):
            perm = rng.sample_int(n)  # R: sample(1:nrow(pm))
            permuted = calculate_metrics(pm_arr[perm, :])["perplexity"].to_numpy()
            counts += permuted <= observed

    p_values = (counts + 1) / (n_perm + 1)
    adata.obs[val_name] = p_values
    return adata


def _resolve_pm(adata, pm, pm_cols):
    if pm is not None:
        return np.asarray(pm, dtype=np.float64)
    if pm_cols is None:
        pm_cols = adata.uns.get("hoodscanr", {}).get("pm_cols")
    if pm_cols is None:
        raise ValueError("Need to input either the pm or pm_cols parameters.")
    return adata.obs.loc[:, list(pm_cols)].to_numpy(dtype=np.float64)
