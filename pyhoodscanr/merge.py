"""Collapse per-neighbour weights onto per-cell-type neighbourhood probabilities.

Mirror of ``hoodscanR/R/merge_by_group.R`` and ``R/merge_pm_spe.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rmath import r_row_sums

__all__ = ["merge_by_group", "merge_hood_adata", "merge_hood_spe"]


def merge_by_group(pm, group_df, continuous_annotation: bool = False) -> pd.DataFrame:
    """``hoodscanR::mergeByGroup`` — sum soft weights within each annotation.

    Parameters
    ----------
    pm
        Probability matrix from :func:`~pyhoodscanr.scan_hoods`
        (``n_cells x k``, rows sum to 1).
    group_df
        * ``continuous_annotation=False`` (default): a ``n_cells x k`` matrix of
          **labels** — the ``"cells"`` frame from
          :func:`~pyhoodscanr.find_near_cells`.
        * ``continuous_annotation=True``: a ``k x n_types`` **numeric** fuzzy
          annotation whose rows sum to 1 (e.g. deconvolution output).
    continuous_annotation
        Interpret ``group_df`` as fuzzy probabilities instead of hard labels.

    Returns
    -------
    pandas.DataFrame
        ``n_cells x n_types``; each row is the cell's neighbourhood
        probability distribution and sums to 1.

    Notes
    -----
    The fuzzy branch has no equivalent in `monkeybread` or `squidpy` — it lets
    label uncertainty propagate into the neighbourhood profile instead of
    being collapsed to an argmax first.
    """
    index = None
    if isinstance(pm, pd.DataFrame):
        index = pm.index
        pm_arr = pm.to_numpy(dtype=np.float64)
    else:
        pm_arr = np.asarray(pm, dtype=np.float64)
    if pm_arr.ndim != 2:
        raise ValueError("The input pm must be a numeric matrix.")

    if not continuous_annotation:
        if isinstance(group_df, pd.DataFrame):
            if index is None:
                index = group_df.index
            g_arr = group_df.to_numpy()
        else:
            g_arr = np.asarray(group_df)
        if g_arr.shape != pm_arr.shape:
            raise ValueError("df and group_df should have the same dimensions.")
        if g_arr.dtype.kind not in "OUS":
            g_arr = g_arr.astype(str)
        # R: uniquegroups <- sort(unique(as.vector(group_df)))
        # pandas.factorize(sort=True) is a hash pass + a sort of the *small*
        # unique set, vs np.unique's full sort of all n*k labels. Identical
        # result (Python's str ordering is code-point order, same as R's sort()
        # in the C locale); ~3x faster. Non-ASCII labels under a non-C locale
        # could collate differently in R -- see MATH.md section 4.
        codes_flat, unique_groups = pd.factorize(g_arr.ravel(), sort=True)
        unique_groups = np.asarray(unique_groups)
        codes = codes_flat.reshape(g_arr.shape)
        # Exact rewrite: masking with 0 and summing in long double is identical
        # to R's rowSums(mask * pm) because x + 0.0 is exact in IEEE-754, so
        # only the order of the *nonzero* terms matters and that is preserved.
        pm_ld = pm_arr.astype(np.longdouble)
        zero = np.longdouble(0.0)
        out = np.empty((pm_arr.shape[0], unique_groups.shape[0]), dtype=np.float64)
        for j in range(unique_groups.shape[0]):
            out[:, j] = np.where(codes == j, pm_ld, zero).sum(axis=1).astype(np.float64)
        columns = list(unique_groups)
    else:
        g_arr = (
            group_df.to_numpy(dtype=np.float64)
            if isinstance(group_df, pd.DataFrame)
            else np.asarray(group_df, dtype=np.float64)
        )
        if not np.issubdtype(g_arr.dtype, np.number):
            raise ValueError(
                "When `continuous_annotation=True`, `group_df` must be numeric."
            )
        if g_arr.shape[0] != pm_arr.shape[1]:
            raise ValueError(
                "For continuous annotations, rows of `group_df` must match columns of `pm`."
            )
        out = pm_arr @ g_arr
        row_sums = r_row_sums(out)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / row_sums[:, None]
        columns = (
            list(group_df.columns)
            if isinstance(group_df, pd.DataFrame)
            else [f"celltype_{i + 1}" for i in range(g_arr.shape[1])]
        )

    out[~np.isfinite(out)] = 0.0  # R: out_df[is.na(out_df)] <- 0
    if index is None:
        index = pd.RangeIndex(out.shape[0])
    return pd.DataFrame(out, index=index, columns=columns)


def merge_hood_adata(adata, pm, val_names=None):
    """``hoodscanR::mergeHoodSpe`` — write the probability matrix into ``adata.obs``.

    Also stores it as ``adata.obsm['hoods']`` and records the column names in
    ``adata.uns['hoodscanr']['pm_cols']`` so downstream calls do not need
    ``pm_cols`` spelled out.
    """
    pm = pd.DataFrame(pm)
    if not pm.index.equals(pd.Index(adata.obs_names)):
        pm = pm.loc[adata.obs_names]

    if val_names is not None:
        if len(val_names) == pm.shape[1]:
            pm.columns = list(val_names)
        else:
            import warnings

            warnings.warn(
                "The length of val_names is not right, using pm columns instead.",
                stacklevel=2,
            )

    for col in pm.columns:
        adata.obs[col] = pm[col].to_numpy()
    adata.obsm["hoods"] = pm.to_numpy(dtype=np.float64)
    adata.uns.setdefault("hoodscanr", {})["pm_cols"] = list(map(str, pm.columns))
    return adata


def merge_hood_spe(adata, pm, val_names=None):
    """Alias for :func:`merge_hood_adata` under the upstream R name.

    ``hoodscanR::mergeHoodSpe`` takes a ``SpatialExperiment``; the AnnData
    equivalent is the only difference. Provided so an R script can be
    transliterated line-for-line.
    """
    return merge_hood_adata(adata, pm, val_names=val_names)
