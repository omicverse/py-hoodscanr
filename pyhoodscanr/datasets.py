"""Canonical fixture loader.

``spe_test`` is hoodscanR's own bundled dataset: a 2661-cell subset of a
NanoString CosMx SMI non-small-cell-lung-cancer section (``Lung9_Rep1``, FOVs
around slide position 5) with six coarse cell-type labels.  Real data, shipped
by the upstream package under GPL-3.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

__all__ = ["load_spe_test", "fixture_path"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = (
    os.path.join(_HERE, "data", "fixture_spe_test.csv"),
    os.path.join(_HERE, "..", "data", "fixture_spe_test.csv"),
)


def fixture_path() -> str:
    """Absolute path to the packaged fixture CSV."""
    for p in _CANDIDATES:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "fixture_spe_test.csv not found; regenerate with "
        "`Rscript tests/export_fixture.R data`."
    )


def load_spe_test(path: str | None = None):
    """Load the canonical fixture as an :class:`anndata.AnnData`.

    Returns
    -------
    anndata.AnnData
        ``obs['cell_annotation']`` holds the six cell types,
        ``obsm['spatial']`` holds the (x, y) coordinates in um.

    Examples
    --------
    >>> import pyhoodscanr as ph
    >>> adata = ph.load_spe_test()
    >>> adata.shape[0]
    2661
    """
    import anndata as ad

    df = pd.read_csv(path or fixture_path(), dtype={"cell_id": str, "cell_annotation": str})
    obs = pd.DataFrame(
        {"cell_annotation": df["cell_annotation"].to_numpy()},
        index=pd.Index(df["cell_id"].to_numpy(), name=None),
    )
    adata = ad.AnnData(
        X=np.ones((df.shape[0], 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["dummy"]),
    )
    adata.obsm["spatial"] = df[["x", "y"]].to_numpy(dtype=np.float64)
    return adata
