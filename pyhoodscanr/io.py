"""Input formatting. Mirror of ``hoodscanR/R/read_data.R``.

``SpatialExperiment`` -> :class:`anndata.AnnData`: ``spatialCoords()`` becomes
``adata.obsm['spatial']`` and the chosen annotation column is normalised to
``adata.obs['cell_annotation']``, exactly as ``readHoodData`` does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["read_hood_data"]


def read_hood_data(
    adata=None,
    anno_col: str | None = None,
    cell_pos_dat: pd.DataFrame | None = None,
    cell_anno_dat: pd.DataFrame | None = None,
    pos_col=None,
    basis: str = "spatial",
    copy: bool = True,
):
    """``hoodscanR::readHoodData``.

    Two calling conventions, mirroring R:

    1. ``read_hood_data(adata, anno_col="celltypes")`` — normalise an existing
       AnnData.
    2. ``read_hood_data(cell_pos_dat=..., cell_anno_dat=...)`` — build one from
       a 3-column position frame (``cell_id, x, y``) and a 2-column annotation
       frame (``cell_id, annotation``).

    Parameters
    ----------
    pos_col
        Two ``obs`` column names to use as coordinates when they are not
        already in ``obsm[basis]``.
    basis
        ``obsm`` key for the coordinates.
    copy
        Return a copy instead of modifying in place.
    """
    import anndata as ad

    if adata is None and anno_col is None:
        if cell_pos_dat is None or cell_anno_dat is None:
            raise ValueError(
                "You need to input either an AnnData with parameter adata "
                "or two DataFrames with parameters cell_pos_dat and cell_anno_dat"
            )
        if not isinstance(cell_pos_dat, pd.DataFrame) or not isinstance(
            cell_anno_dat, pd.DataFrame
        ):
            raise ValueError("cell_pos_dat and cell_anno_dat should be DataFrames.")
        if cell_pos_dat.shape[1] != 3:
            raise ValueError(
                "The cell_pos_dat is expected to have three columns: cell_id, x, and y."
            )
        if cell_anno_dat.shape[1] != 2:
            raise ValueError(
                "The cell_anno_dat is expected to have two columns: cell_id and annotations."
            )
        if cell_pos_dat.shape[0] != cell_anno_dat.shape[0]:
            raise ValueError(
                "The cell_pos_dat should have the same amount of cells as cell_anno_dat."
            )
        pos = cell_pos_dat.copy()
        pos.columns = ["cell_id", "x", "y"]
        ann = cell_anno_dat.copy()
        ann.columns = ["cell_id", "cell_annotation"]
        obs = pd.DataFrame(
            {"cell_annotation": ann["cell_annotation"].astype(str).to_numpy()},
            index=pd.Index(pos["cell_id"].astype(str), name=None),
        )
        out = ad.AnnData(
            X=np.ones((pos.shape[0], 10), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=[f"gene_{i + 1}" for i in range(10)]),
        )
        out.obsm[basis] = pos[["x", "y"]].to_numpy(dtype=np.float64)
        return out

    if adata is None:
        raise ValueError("adata must be an AnnData and anno_col a column of adata.obs.")
    if anno_col is None or anno_col not in adata.obs.columns:
        raise ValueError("anno_col is not in adata.obs.")

    out = adata.copy() if copy else adata
    out.obs["cell_annotation"] = out.obs[anno_col].astype(str).to_numpy()

    if pos_col is not None:
        if not all(c in out.obs.columns for c in pos_col):
            raise ValueError("pos_col is not in adata.obs.")
        out.obsm[basis] = out.obs.loc[:, list(pos_col)].to_numpy(dtype=np.float64)
    elif basis not in out.obsm:
        raise ValueError(
            f"Coordinates are not found in adata.obsm['{basis}']; pass pos_col instead."
        )
    return out
