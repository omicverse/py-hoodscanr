"""py-hoodscanR — pure-Python mirror of the Bioconductor package `hoodscanR`.

Spatial cellular *neighbourhood* scanning: for every cell, a distance-weighted
probability distribution over the cell types around it, plus the entropy and
perplexity of that distribution.

Quick start
-----------
>>> import pyhoodscanr as ph
>>> adata = ph.load_spe_test()
>>> hs = ph.HoodScanR(adata, anno_col="cell_annotation")
>>> _ = hs.find_near_cells(k=100).scan_hoods().merge_by_group().calc_metrics()
>>> "perplexity" in hs.adata.obs
True

Upstream: https://github.com/DavisLaboratory/hoodscanR
Paper: https://doi.org/10.1101/2024.03.26.586902
"""

from ._kmeans_hw import kmeans_hartigan_wong, r_kmeans
from ._roptim import r_optim_bfgs
from ._rrng import RRandom
from .clustering import clust_by_hood
from .core import HoodScanR
from .datasets import fixture_path, load_spe_test
from .io import read_hood_data
from .merge import merge_by_group, merge_hood_adata, merge_hood_spe
from .metrics import calc_metrics, calculate_metrics, perplexity_permute
from .neighbours import find_near_cells
from .plotting import (
    plot_colocal,
    plot_hood_mat,
    plot_prob_dist,
    plot_tissue,
)
from .soft_neighbourhood import f_nll, scan_hoods, soft_max_intl

__version__ = "0.1.0"

__all__ = [
    "HoodScanR",
    # R-mirror functional API
    "read_hood_data",
    "find_near_cells",
    "scan_hoods",
    "merge_by_group",
    "merge_hood_adata",
    "merge_hood_spe",
    "calc_metrics",
    "calculate_metrics",
    "perplexity_permute",
    "clust_by_hood",
    "plot_tissue",
    "plot_hood_mat",
    "plot_prob_dist",
    "plot_colocal",
    # internals worth exposing
    "soft_max_intl",
    "f_nll",
    "kmeans_hartigan_wong",
    "r_kmeans",
    "r_optim_bfgs",
    "RRandom",
    # data
    "load_spe_test",
    "fixture_path",
    "__version__",
]
