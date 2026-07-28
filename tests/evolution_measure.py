"""Re-measure the pipeline under each Acceleration iteration's configuration.

Every iteration is re-run from the same code base with the relevant fast path
disabled, so the numbers in ITERATION_LOG.md are measured, not reconstructed.

    python tests/evolution_measure.py            -> data/evolution_runs.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import pyhoodscanr as ph  # noqa: E402
from pyhoodscanr import _rrng_fast, merge as merge_mod, metrics as metrics_mod  # noqa: E402
from pyhoodscanr._rmath import r_row_sums  # noqa: E402
from parity import evaluate, load_manifest, load_reference  # noqa: E402

N_PERM = 1000


# --- iteration-0 implementations (literal transcription of the R source) ----
def merge_by_group_v0(pm, group_df, continuous_annotation=False):
    """Literal transcription: one boolean mask per group, R rowSums."""
    index = group_df.index if isinstance(group_df, pd.DataFrame) else None
    pm_arr = np.asarray(pm, dtype=np.float64)
    g = (group_df.to_numpy() if isinstance(group_df, pd.DataFrame)
         else np.asarray(group_df)).astype(str)
    unique_groups = np.array(sorted(set(g.ravel().tolist())), dtype=object)
    out = np.empty((pm_arr.shape[0], unique_groups.shape[0]), dtype=np.float64)
    for j, gg in enumerate(unique_groups):
        out[:, j] = r_row_sums((g == gg).astype(np.float64) * pm_arr)
    out[~np.isfinite(out)] = 0.0
    return pd.DataFrame(out, index=index if index is not None
                        else pd.RangeIndex(out.shape[0]), columns=list(unique_groups))


def run_pipeline(merge_fn, permute_kwargs):
    adata = ph.load_spe_test()
    fnc = ph.find_near_cells(adata, k=100)
    pm = ph.scan_hoods(fnc["distance"].to_numpy(), verbose=False)
    hoods = merge_fn(pm, fnc["cells"])
    hoods.index = adata.obs_names
    ph.merge_hood_adata(adata, hoods)
    H = hoods.to_numpy()
    ph.calc_metrics(adata, pm=H)
    ph.perplexity_permute(adata, pm=H, n_perm=N_PERM, seed=42, **permute_kwargs)
    ph.clust_by_hood(adata, pm_cols=list(hoods.columns), k=10, seed=42)
    colocal = ph.plot_colocal(adata, pm_cols=list(hoods.columns), return_matrix=True)
    return {
        "hoods": H,
        "entropy": adata.obs["entropy"].to_numpy(),
        "perplexity": adata.obs["perplexity"].to_numpy(),
        "perplexity_p": adata.obs["perplexity_p"].to_numpy(),
        "clusters": adata.obs["clusters"].astype(int).to_numpy(),
        "colocal": colocal.to_numpy(),
        "distance": fnc["distance"].to_numpy(),
        "pm": pm,
    }


def measure(label, merge_fn, permute_kwargs, force_no_numba, n_runs=3):
    saved = _rrng_fast.HAVE_NUMBA
    saved_m = metrics_mod  # noqa: F841
    if force_no_numba:
        _rrng_fast.HAVE_NUMBA = False
    try:
        t0 = time.perf_counter()
        res = run_pipeline(merge_fn, permute_kwargs)
        warm = time.perf_counter() - t0
        runs = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            res = run_pipeline(merge_fn, permute_kwargs)
            runs.append(time.perf_counter() - t0)
    finally:
        _rrng_fast.HAVE_NUMBA = saved

    ref = load_reference()
    man = load_manifest()
    # candidate keys the evaluator expects
    cand = dict(res)
    cand["tau_smooth"] = np.float64(ref["tau_smooth"])  # unchanged by these rewrites
    rows = evaluate(ref, cand, man)
    by = {r["name"]: r for r in rows}
    print(f"{label:34s} {statistics.fmean(runs):8.3f}s +- {statistics.stdev(runs) if len(runs)>1 else 0:.3f}"
          f"   cosine={by['pm_merged_cosine']['value']:.12f}"
          f"   pass={sum(r['pass'] for r in rows)}/{len(rows)}")
    return {
        "label": label,
        "warmup_run_s": warm,
        "wall_clock_runs_s": runs,
        "wall_clock_mean_s": statistics.fmean(runs),
        "wall_clock_stddev_s": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "parity": {r["name"]: r["value"] for r in rows},
        "parity_passes": {r["name"]: r["pass"] for r in rows},
        "n_pass": int(sum(r["pass"] for r in rows)),
        "n_total": len(rows),
    }


def main():
    out = {}
    out["iter0"] = measure(
        "iter0 literal transcription", merge_by_group_v0,
        {"exact": False}, force_no_numba=True)
    out["iter1"] = measure(
        "iter1 +permutation identity", merge_by_group_v0,
        {"exact": True}, force_no_numba=True)
    out["iter2"] = measure(
        "iter2 +compiled R RNG", merge_by_group_v0,
        {"exact": True}, force_no_numba=False)
    out["iter3"] = measure(
        "iter3 +factorised merge", ph.merge_by_group,
        {"exact": True}, force_no_numba=False)
    with open(os.path.join(ROOT, "data", "evolution_runs.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote data/evolution_runs.json")
    return out


if __name__ == "__main__":
    main()
