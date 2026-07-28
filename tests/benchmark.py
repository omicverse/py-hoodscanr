"""Wall-clock benchmark: py-hoodscanR vs hoodscanR, per pipeline stage.

    python tests/benchmark.py [n_repeat]

Warm-up run discarded (numba JIT / BLAS thread spin-up); reports mean +- sd
over ``n_repeat`` further runs.  Writes ``data/benchmark.json``.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import pyhoodscanr as ph  # noqa: E402
from pyhoodscanr._rmath import r_median  # noqa: E402


def _time(fn, n_repeat):
    fn()  # warm-up, discarded
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"mean": statistics.fmean(ts), "sd": statistics.stdev(ts) if len(ts) > 1 else 0.0}


def python_timings(n_repeat=3):
    adata = ph.load_spe_test()
    out = {}
    out["find_near_cells"] = _time(lambda: ph.find_near_cells(adata, k=100), n_repeat)

    fnc = ph.find_near_cells(adata, k=100)
    dist = fnc["distance"].to_numpy()
    out["scan_hoods"] = _time(lambda: ph.scan_hoods(dist, verbose=False), n_repeat)

    sub = dist[:500]
    out["scan_hoods_smooth"] = _time(
        lambda: ph.scan_hoods(sub, mode="smoothFadeout", verbose=False), n_repeat
    )

    pm = ph.scan_hoods(dist, verbose=False)
    out["merge_by_group"] = _time(lambda: ph.merge_by_group(pm, fnc["cells"]), n_repeat)

    hoods = ph.merge_by_group(pm, fnc["cells"])
    hoods.index = adata.obs_names
    ph.merge_hood_adata(adata, hoods)
    H = hoods.to_numpy()
    out["calc_metrics"] = _time(lambda: ph.calc_metrics(adata, pm=H), n_repeat)
    out["perplexity_permute"] = _time(
        lambda: ph.perplexity_permute(adata, pm=H, n_perm=1000, seed=42), n_repeat
    )
    out["clust_by_hood"] = _time(
        lambda: ph.clust_by_hood(adata, pm_cols=list(hoods.columns), k=10, seed=42),
        n_repeat,
    )
    return out


def r_timings(n_repeat=3):
    rscript = os.environ.get(
        "RSCRIPT", "/scratch/users/steorra/env/hoodR/bin/Rscript"
    )
    if not os.path.exists(rscript):
        return None
    runs = []
    for _ in range(n_repeat + 1):
        tmp = os.path.join(ROOT, "data", "_bench_ref.json")
        subprocess.run(
            [rscript, os.path.join(HERE, "r_reference_driver.R"),
             os.path.join(ROOT, "data", "fixture_spe_test.csv"), tmp],
            check=True, capture_output=True,
        )
        with open(tmp) as fh:
            runs.append(json.load(fh)["timings"])
    runs = runs[1:]  # discard warm-up
    keys = runs[0].keys()
    return {
        k: {"mean": statistics.fmean([float(np.ravel(r[k])[0]) for r in runs]),
            "sd": statistics.stdev([float(np.ravel(r[k])[0]) for r in runs]) if len(runs) > 1 else 0.0}
        for k in keys
    }


def main(n_repeat=3):
    py = python_timings(n_repeat)
    r = r_timings(n_repeat)
    res = {"python": py, "r": r, "n_repeat": n_repeat}
    with open(os.path.join(ROOT, "data", "benchmark.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"{'stage':24s}{'R (s)':>18s}{'Python (s)':>18s}{'speedup':>10s}")
    print("-" * 70)
    tot_r = tot_p = 0.0
    for k in py:
        p = py[k]
        rr = r[k] if r and k in r else None
        tot_p += p["mean"]
        if rr:
            tot_r += rr["mean"]
            print(f"{k:24s}{rr['mean']:12.4f}+-{rr['sd']:<4.3f}"
                  f"{p['mean']:12.4f}+-{p['sd']:<4.3f}{rr['mean'] / p['mean']:9.2f}x")
        else:
            print(f"{k:24s}{'n/a':>18s}{p['mean']:12.4f}+-{p['sd']:<4.3f}{'':>10s}")
    print("-" * 70)
    print(f"{'TOTAL':24s}{tot_r:18.4f}{tot_p:18.4f}{tot_r / tot_p:9.2f}x")
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
