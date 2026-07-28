"""Pack the R reference into a single compressed .npz small enough to commit.

    python tests/pack_reference.py

Reads ``data/reference_output.json`` + ``data/reference_output_bin/`` (produced
by ``tests/r_reference_driver.R``) and writes ``data/reference_output.npz``.
Every array named in ``data/manifest.yaml::outputs[]`` is preserved at full f64
precision; only the JSON text form is dropped.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from parity import load_reference  # noqa: E402

KEEP = [
    "cell_id", "celltypes", "k", "distance", "cells", "tau_prox", "pm",
    "hoods", "hoods_fuzzy", "fuzzy_input", "sub_idx", "tau_smooth",
    "nll_smooth", "nll_at_init", "tau_smooth_init", "pm_smooth",
    "entropy", "perplexity", "perplexity_p", "n_perm", "clusters",
    "k_clust", "centroids", "colocal", "mean_by_group", "mean_by_group_rows",
]


def main():
    ref = load_reference(os.path.join(ROOT, "data", "reference_output.json"))
    payload = {}
    for k in KEEP:
        if k not in ref:
            print(f"  (missing: {k})")
            continue
        v = ref[k]
        payload[k] = np.asarray(v) if not np.isscalar(v) else np.asarray(v)
    out = os.path.join(ROOT, "data", "reference_output.npz")
    np.savez_compressed(out, **payload)
    print(f"wrote {out}  {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
