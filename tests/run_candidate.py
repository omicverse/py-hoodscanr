"""Run the Python port on the canonical fixture and dump the gated outputs.

    python tests/run_candidate.py data/fixture_spe_test.csv data/candidate_output.npz

Mirrors ``tests/r_reference_driver.R`` call-for-call so the two JSON/NPZ
payloads can be diffed field-by-field.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import pyhoodscanr as ph
from pyhoodscanr._rrng import RRandom
from pyhoodscanr._roptim import r_optim_bfgs
from pyhoodscanr.soft_neighbourhood import f_nll
from pyhoodscanr._rmath import r_median


def main(fixture_path, out_path, k=100, n_perm=1000, k_clust=10):
    t = {}
    adata = ph.load_spe_test(fixture_path)

    t0 = time.perf_counter()
    fnc = ph.find_near_cells(adata, k=k)
    t["find_near_cells"] = time.perf_counter() - t0

    dist = fnc["distance"].to_numpy()

    t0 = time.perf_counter()
    pm = ph.scan_hoods(dist, verbose=False)
    t["scan_hoods"] = time.perf_counter() - t0
    tau_prox = r_median(dist**2) / 5

    # smoothFadeout on the same deterministic subsample the R driver uses
    sub = dist[:500]
    t0 = time.perf_counter()
    pm_smooth, tau_smooth = ph.scan_hoods(sub, mode="smoothFadeout", verbose=False,
                                          return_tau=True)
    t["scan_hoods_smooth"] = time.perf_counter() - t0
    t_init = r_median(sub**2)
    opt = r_optim_bfgs(np.array([t_init]), lambda p: f_nll(sub, p[0]))
    nll_smooth = opt.value
    nll_at_init = f_nll(sub, t_init)

    t0 = time.perf_counter()
    hoods = ph.merge_by_group(pm, fnc["cells"])
    t["merge_by_group"] = time.perf_counter() - t0
    hoods.index = adata.obs_names

    # fuzzy branch — fuzzy matrix comes from the R driver (set.seed(42); runif)
    fuzzy = _r_runif_matrix(pm.shape[1], 3, seed=42)
    hoods_fuzzy = ph.merge_by_group(pm, fuzzy, continuous_annotation=True)

    ph.merge_hood_adata(adata, hoods)

    t0 = time.perf_counter()
    ph.calc_metrics(adata, pm=hoods.to_numpy())
    t["calc_metrics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ph.perplexity_permute(adata, pm=hoods.to_numpy(), n_perm=n_perm, seed=42)
    t["perplexity_permute"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ph.clust_by_hood(adata, pm_cols=list(hoods.columns), k=k_clust, seed=42)
    t["clust_by_hood"] = time.perf_counter() - t0

    colocal = ph.plot_colocal(adata, pm_cols=list(hoods.columns), return_matrix=True)
    mean_by_group = ph.plot_colocal(adata, pm_cols=list(hoods.columns),
                                    self_cor=False, by_group="cell_annotation",
                                    return_matrix=True)

    np.savez_compressed(
        out_path,
        cell_id=np.asarray(adata.obs_names, dtype=object),
        celltypes=np.asarray(list(hoods.columns), dtype=object),
        distance=dist,
        cells=fnc["cells"].to_numpy().astype(object),
        tau_prox=np.float64(tau_prox),
        pm=pm,
        hoods=hoods.to_numpy(),
        hoods_fuzzy=hoods_fuzzy.to_numpy(),
        fuzzy_input=fuzzy,
        tau_smooth=np.float64(tau_smooth),
        nll_smooth=np.float64(nll_smooth),
        nll_at_init=np.float64(nll_at_init),
        tau_smooth_init=np.float64(t_init),
        pm_smooth=pm_smooth,
        entropy=adata.obs["entropy"].to_numpy(),
        perplexity=adata.obs["perplexity"].to_numpy(),
        perplexity_p=adata.obs["perplexity_p"].to_numpy(),
        clusters=adata.obs["clusters"].astype(int).to_numpy(),
        centroids=adata.uns["hoodscanr"]["centroids"],
        colocal=colocal.to_numpy(),
        mean_by_group=mean_by_group.to_numpy(),
        mean_by_group_rows=np.asarray(list(mean_by_group.index), dtype=object),
        timings=np.asarray([t[key] for key in sorted(t)], dtype=np.float64),
        timing_names=np.asarray(sorted(t), dtype=object),
        allow_pickle=True,
    )
    print("wrote", out_path)
    for kk, vv in t.items():
        print(f"  {kk:22s} {vv:8.3f}s")


def _r_runif_matrix(nrow, ncol, seed=42):
    """R: set.seed(42); matrix(runif(nrow*ncol), ncol=ncol) then row-normalise.

    R fills matrices column-major.
    """
    rng = RRandom(seed)
    vals = np.array([rng.unif_rand() for _ in range(nrow * ncol)])
    m = vals.reshape((nrow, ncol), order="F")
    return m / m.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
