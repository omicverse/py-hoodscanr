"""Method-chaining class API over the R-mirror functions."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .clustering import clust_by_hood
from .io import read_hood_data
from .merge import merge_by_group, merge_hood_adata
from .metrics import calc_metrics, perplexity_permute
from .neighbours import find_near_cells

__all__ = ["HoodScanR"]


class HoodScanR:
    """Chainable wrapper around the hoodscanR pipeline.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with coordinates in ``obsm[basis]``.
    anno_col
        Cell-type annotation column in ``obs``.
    basis
        ``obsm`` key for the spatial coordinates.

    Examples
    --------
    >>> import pyhoodscanr as ph
    >>> hs = ph.HoodScanR(ph.load_spe_test(), anno_col="cell_annotation")
    >>> _ = (hs.find_near_cells(k=100)
    ...        .scan_hoods()
    ...        .merge_by_group()
    ...        .calc_metrics()
    ...        .clust_by_hood(k=10))
    >>> sorted(hs.hoods.columns)[:2]
    ['Dividing.cells', 'Endothelial.cell']
    """

    def __init__(self, adata, anno_col: str = "cell_annotation", basis: str = "spatial"):
        self.adata = read_hood_data(adata, anno_col=anno_col, basis=basis, copy=True)
        self.basis = basis
        self.fnc = None
        self.pm = None
        self.tau = None
        self.hoods = None

    # ------------------------------------------------------------------
    def find_near_cells(self, k: int = 100, report_cell_id: bool = False, **kwargs):
        """See :func:`pyhoodscanr.find_near_cells`."""
        self.fnc = find_near_cells(
            self.adata, k=k, report_cell_id=report_cell_id, report_dist=True,
            basis=self.basis, **kwargs
        )
        return self

    def scan_hoods(
        self,
        mode: Literal["proximityFocused", "smoothFadeout"] = "proximityFocused",
        tau: float | None = None,
        t_init: float | None = None,
        verbose: bool = False,
    ):
        """See :func:`pyhoodscanr.scan_hoods`."""
        from .soft_neighbourhood import scan_hoods as _scan

        if self.fnc is None:
            raise RuntimeError("Call find_near_cells() first.")
        self.pm, self.tau = _scan(
            self.fnc["distance"].to_numpy(), mode=mode, tau=tau, t_init=t_init,
            verbose=verbose, return_tau=True,
        )
        self.adata.uns.setdefault("hoodscanr", {})["tau"] = self.tau
        self.adata.uns["hoodscanr"]["mode"] = mode
        return self

    def merge_by_group(self, group_df=None, continuous_annotation: bool = False):
        """See :func:`pyhoodscanr.merge_by_group`; also writes into ``adata``."""
        if self.pm is None:
            raise RuntimeError("Call scan_hoods() first.")
        if group_df is None:
            group_df = self.fnc["cells"]
        self.hoods = merge_by_group(
            self.pm, group_df, continuous_annotation=continuous_annotation
        )
        self.hoods.index = self.adata.obs_names
        merge_hood_adata(self.adata, self.hoods)
        return self

    def calc_metrics(self, val_names=("entropy", "perplexity")):
        """See :func:`pyhoodscanr.calc_metrics`."""
        calc_metrics(self.adata, pm=np.asarray(self.hoods), val_names=val_names)
        return self

    def perplexity_permute(self, n_perm: int = 1000, seed: int = 42, **kwargs):
        """See :func:`pyhoodscanr.perplexity_permute`."""
        perplexity_permute(
            self.adata, pm=np.asarray(self.hoods), n_perm=n_perm, seed=seed, **kwargs
        )
        return self

    def clust_by_hood(self, k: int = 0, nstart: int = 5, iter_max: int = 1000,
                      seed: int = 42, val_name: str = "clusters"):
        """See :func:`pyhoodscanr.clust_by_hood`."""
        clust_by_hood(
            self.adata, pm_cols=list(self.hoods.columns), k=k, nstart=nstart,
            iter_max=iter_max, seed=seed, val_name=val_name,
        )
        return self

    # ------------------------------------------------------------------
    def run(self, k: int = 100, mode="proximityFocused", n_clusters: int = 0,
            n_perm: int | None = None, seed: int = 42):
        """Run the whole upstream vignette pipeline in one call."""
        self.find_near_cells(k=k).scan_hoods(mode=mode).merge_by_group().calc_metrics()
        if n_perm:
            self.perplexity_permute(n_perm=n_perm, seed=seed)
        self.clust_by_hood(k=n_clusters, seed=seed)
        return self

    def __repr__(self):  # pragma: no cover - cosmetic
        n = self.adata.n_obs
        t = "-" if self.tau is None else f"{self.tau:.6g}"
        h = "-" if self.hoods is None else f"{self.hoods.shape[1]}"
        return f"HoodScanR(n_cells={n}, tau={t}, n_hoods={h})"
