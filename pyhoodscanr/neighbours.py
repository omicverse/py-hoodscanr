"""k-nearest-cell search. Mirror of ``hoodscanR/R/find_near_cells.R``."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

__all__ = ["find_near_cells"]


def find_near_cells(
    adata,
    k: int = 100,
    target_cell: Sequence[str] | None = None,
    report_cell_id: bool = False,
    report_dist: bool = True,
    anno_col: str = "cell_annotation",
    basis: str = "spatial",
    tie_break: str = "stable",
    warn_boundary_ties: bool = False,
):
    """``hoodscanR::findNearCells`` — the k nearest cells of every cell.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with coordinates in ``adata.obsm[basis]``
        (the AnnData analogue of ``SpatialExperiment``'s ``spatialCoords``).
    k
        Number of nearest cells to return.  R default 100.
    target_cell
        Restrict the query set to these cells (R's ``targetCell``).
        ``None`` (R's ``FALSE``) queries every cell.
    report_cell_id
        Report neighbour **cell ids** instead of neighbour **cell types**.
    report_dist
        Also return the distance matrix.
    anno_col
        Column of ``adata.obs`` holding the annotation.
    basis
        Key of ``adata.obsm`` holding the coordinates.
    tie_break
        ``"stable"`` (default) sorts each row by ``(distance, cell index)``,
        which makes the output independent of the search backend and stable
        across platforms.  ``"backend"`` keeps scikit-learn's raw order.
    warn_boundary_ties
        Warn when a cell's k-th and (k+1)-th neighbours are exactly
        equidistant, i.e. when the *membership* of the neighbourhood (not just
        its order) is ambiguous.

    Returns
    -------
    dict
        ``{"cells": DataFrame(n x k), "distance": DataFrame(n x k)}`` when
        ``report_dist`` is True, else just the ``cells`` DataFrame.

    Notes
    -----
    R searches with ``RANN::nn2(..., k = k + 1, searchtype = "priority")`` and
    then **drops the first column**, which is assumed to be the query point
    itself.  We reproduce that exactly, including the assumption.  ``RANN``'s
    default ``eps = 0`` makes the ANN search exact, so the neighbour *distances*
    agree with scikit-learn's exact k-d tree to f64.

    Equidistant neighbours are a different matter.  ANN's priority queue and
    scikit-learn's heap order ties differently, and neither order is part of
    the algorithm's specification.  Reordering ties is harmless downstream —
    :func:`~pyhoodscanr.merge_by_group` sums softmax weights per label, and
    equal distances carry equal weights, so the per-label sums are invariant.
    The one case that is *not* invariant is a tie straddling the k-th
    position, where the two implementations include different cells in the
    neighbourhood.  ``warn_boundary_ties=True`` reports those cells.
    """
    coords = np.asarray(adata.obsm[basis], dtype=np.float64)
    if coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{basis}'] must have at least 2 columns.")
    coords = coords[:, :2]

    cell_ids = np.asarray(adata.obs_names, dtype=object)
    if anno_col not in adata.obs.columns:
        raise ValueError(f"anno_col '{anno_col}' is not in adata.obs.")
    cell_annotation = np.asarray(adata.obs[anno_col].astype(str), dtype=object)

    if target_cell is None:
        query = coords
        query_names = cell_ids
    else:
        target_cell = np.atleast_1d(np.asarray(target_cell, dtype=object))
        pos = pd.Index(adata.obs_names).get_indexer(target_cell)
        if (pos < 0).any():
            missing = target_cell[pos < 0]
            raise KeyError(f"target_cell not found in adata.obs_names: {list(missing)}")
        query = coords[pos]
        query_names = target_cell

    n_fetch = min(k + 2, coords.shape[0]) if tie_break == "stable" else k + 1
    nn = NearestNeighbors(n_neighbors=n_fetch, algorithm="kd_tree", metric="euclidean")
    nn.fit(coords)
    dists, idx = nn.kneighbors(query, n_neighbors=n_fetch)

    if tie_break == "stable":
        # canonical order: ascending distance, then ascending cell index
        order = np.lexsort((idx, dists), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dists = np.take_along_axis(dists, order, axis=1)
        if warn_boundary_ties and dists.shape[1] > k + 1:
            tied = np.flatnonzero(dists[:, k] == dists[:, k + 1])
            if tied.size:
                import warnings

                warnings.warn(
                    f"{tied.size} cell(s) have an exact distance tie at the k-th "
                    "neighbour; which of the tied cells enters the neighbourhood is "
                    "arbitrary (R's ANN and scikit-learn may disagree). "
                    f"First few: {list(np.asarray(query_names)[tied[:5]])}",
                    stacklevel=2,
                )
    elif tie_break != "backend":
        raise ValueError("tie_break must be 'stable' or 'backend'")

    # R: closest[, 1] is the self hit; neighbours are columns 2..k+1
    idx_close = idx[:, 1 : k + 1]
    dist_close = dists[:, 1 : k + 1]

    if report_cell_id:
        values = cell_ids[idx_close]
    else:
        values = cell_annotation[idx_close]

    colnames = [f"nearest_cell_{i}" for i in range(1, k + 1)]
    cells = pd.DataFrame(values, index=pd.Index(query_names, name=None), columns=colnames)

    if not report_dist:
        return cells

    distance = pd.DataFrame(dist_close, index=cells.index, columns=colnames)
    return {"cells": cells, "distance": distance}
