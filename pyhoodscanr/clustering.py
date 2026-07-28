"""Neighbourhood-based clustering. Mirror of ``hoodscanR/R/pm_clust.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._kmeans_hw import r_kmeans

__all__ = ["clust_by_hood"]


def clust_by_hood(
    object,
    pm_cols=None,
    k: int = 0,
    iter_max: int = 1000,
    nstart: int = 5,
    algo: str = "Hartigan-Wong",
    val_name: str = "clusters",
    seed: int = 42,
):
    """``hoodscanR::clustByHood`` — k-means on the neighbourhood probabilities.

    Parameters
    ----------
    object
        Either a probability matrix (``ndarray`` / ``DataFrame``) or an
        :class:`anndata.AnnData` carrying the probabilities in ``obs``.
    pm_cols
        Required for the AnnData path (falls back to
        ``adata.uns['hoodscanr']['pm_cols']``).
    k
        Number of clusters.  ``0`` (R's default for the AnnData/SPE method)
        means ``2**n_celltypes - 1``.
    iter_max, nstart, algo
        Passed through to k-means.  Only ``"Hartigan-Wong"`` (R's default) is
        implemented; ``"Lloyd"``/``"MacQueen"`` raise.
    val_name
        ``adata.obs`` column to write the cluster labels to.
    seed
        Behaves exactly like ``set.seed(seed)`` before ``kmeans`` in R.

    Returns
    -------
    The input AnnData (labels in ``obs[val_name]``, centroids in
    ``uns['hoodscanr']['centroids']``), or a result dict for the matrix path.

    Notes
    -----
    Cluster labels are 1-based strings, matching R.
    """
    if algo != "Hartigan-Wong":
        raise NotImplementedError(
            f"algo='{algo}' is not ported; hoodscanR's default is 'Hartigan-Wong'."
        )

    is_adata = hasattr(object, "obs") and hasattr(object, "obs_names")

    if is_adata:
        adata = object
        if pm_cols is None:
            pm_cols = adata.uns.get("hoodscanr", {}).get("pm_cols")
        if pm_cols is None:
            raise ValueError("The pm_cols are not included in the AnnData.")
        missing = [c for c in pm_cols if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"The pm_cols are not included in the AnnData: {missing}")
        dat = adata.obs.loc[:, list(pm_cols)].to_numpy(dtype=np.float64)
    else:
        dat = (
            object.to_numpy(dtype=np.float64)
            if isinstance(object, pd.DataFrame)
            else np.asarray(object, dtype=np.float64)
        )

    if k == 0:
        k = 2 ** dat.shape[1] - 1

    res = r_kmeans(dat, k=k, iter_max=iter_max, nstart=nstart, seed=seed)

    if not is_adata:
        return res

    adata.obs[val_name] = pd.Categorical((res["cluster"] + 1).astype(str))
    adata.uns.setdefault("hoodscanr", {})["centroids"] = res["centers"]
    adata.uns["hoodscanr"]["wss"] = res["wss"]
    return adata
