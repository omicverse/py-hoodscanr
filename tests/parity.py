"""Class-aware parity metrics + the gate evaluation, shared by the pytest gate
and the notebooks."""

from __future__ import annotations

import json
import os

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MANIFEST = os.path.join(ROOT, "data", "manifest.yaml")


def load_manifest(path: str = MANIFEST) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_reference(path: str | None = None) -> dict:
    """Load the R reference.

    Prefers the committed compressed ``reference_output.npz`` (4.6 MB, full f64
    precision). Falls back to the raw ``reference_output.json`` +
    ``reference_output_bin/`` pair emitted by ``tests/r_reference_driver.R``
    (22 MB, regenerable, not committed).
    """
    if path is None:
        npz = os.path.join(ROOT, "data", "reference_output.npz")
        js = os.path.join(ROOT, "data", "reference_output.json")
        if os.path.exists(npz) and not os.path.exists(js):
            z = np.load(npz, allow_pickle=True)
            return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        path = js
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for k, v in raw.items():
        if isinstance(v, list):
            try:
                out[k] = np.array(v)
            except Exception:  # ragged
                out[k] = v
        else:
            out[k] = v
    # numeric arrays are shipped as raw little-endian f64 (JSON truncates to
    # ~15 significant digits, which would fake ~1e-12 of error into a 1e-8 gate)
    shapes = out.pop("_bin_shapes", None)
    bindir = out.pop("_bin_dir", None)
    if shapes is not None and bindir is not None:
        base = os.path.join(os.path.dirname(os.path.abspath(path)), str(bindir))
        for name, shape in shapes.items():
            shape = [int(s) for s in np.atleast_1d(shape)]
            arr = np.fromfile(os.path.join(base, f"{name}.f64"), dtype="<f8")
            out[name] = arr.reshape(shape) if shape[1] != 1 else arr.reshape(shape[0])
    return out


def load_candidate(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "data", "candidate_output.npz")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def max_abs_err(ref, cand) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    return float(np.max(np.abs(ref - cand)))


def mean_per_cell_cosine(ref, cand) -> float:
    """Mean over rows of ``cos(p_R[i], p_py[i])`` — the distributional gate."""
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    num = (ref * cand).sum(axis=1)
    den = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den > 0, num / den, 1.0)
    return float(np.mean(np.clip(cos, -1.0, 1.0)))


def max_per_cell_total_variation(ref, cand) -> float:
    """``max_i 0.5 * sum_j |p_R[i,j] - p_py[i,j]|``."""
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    return float(np.max(0.5 * np.abs(ref - cand).sum(axis=1)))


def relative_error(ref, cand) -> float:
    ref = float(np.asarray(ref).ravel()[0])
    cand = float(np.asarray(cand).ravel()[0])
    denom = abs(ref) if abs(ref) > 0 else 1.0
    return abs(ref - cand) / denom


def pearson(ref, cand) -> float:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    cand = np.asarray(cand, dtype=np.float64).ravel()
    if np.allclose(ref, ref[0]) and np.allclose(cand, cand[0]):
        return 1.0
    return float(pearsonr(ref, cand)[0])


def spearman(ref, cand) -> float:
    return float(spearmanr(np.asarray(ref).ravel(), np.asarray(cand).ravel())[0])


def ari(ref, cand) -> float:
    return float(adjusted_rand_score(np.asarray(ref).ravel(), np.asarray(cand).ravel()))


_METRICS = {
    "deterministic": max_abs_err,
    "mean_per_cell_cosine": mean_per_cell_cosine,
    "max_per_cell_total_variation": max_per_cell_total_variation,
    "relative_error": relative_error,
    "pearson": pearson,
    "spearman": spearman,
    "ari": ari,
}

_LOWER_IS_BETTER = {"deterministic", "max_per_cell_total_variation", "relative_error"}


def evaluate(reference: dict, candidate: dict, manifest: dict) -> list[dict]:
    """Evaluate every ``outputs[]`` block. Returns one row per gated output."""
    rows = []
    for spec in manifest["outputs"]:
        name = spec["name"]
        key = spec["location_reference"].lstrip("$.")
        if key not in reference:
            rows.append({"name": name, "metric": spec["metric"], "value": None,
                         "threshold": spec["threshold"], "pass": False,
                         "note": f"missing '{key}' in reference"})
            continue
        cand_key = _CANDIDATE_KEY.get(name, key)
        if cand_key not in candidate:
            rows.append({"name": name, "metric": spec["metric"], "value": None,
                         "threshold": spec["threshold"], "pass": False,
                         "note": f"missing '{cand_key}' in candidate"})
            continue
        fn = _METRICS[spec["metric"]]
        value = fn(reference[key], candidate[cand_key])
        thr = float(spec["threshold"])
        if spec["metric"] in _LOWER_IS_BETTER:
            ok = value <= thr
        else:
            ok = value >= thr - 1e-12  # pearsonr is itself f64-noisy
        rows.append({"name": name, "metric": spec["metric"], "value": value,
                     "threshold": thr, "pass": bool(ok), "note": spec.get("note", "")})
    return rows


# outputs whose candidate array is stored under a different key than the
# reference JSON path
_CANDIDATE_KEY = {
    "knn_distance": "distance",
    "pm_raw": "pm",
    "pm_merged_cosine": "hoods",
    "pm_merged_tv": "hoods",
}


def format_table(rows) -> str:
    w = max(len(r["name"]) for r in rows) + 2
    out = [f"{'output'.ljust(w)}{'metric':<30}{'value':>16}{'threshold':>14}   verdict"]
    out.append("-" * (w + 76))
    for r in rows:
        val = "n/a" if r["value"] is None else f"{r['value']:.6g}"
        out.append(
            f"{r['name'].ljust(w)}{r['metric']:<30}{val:>16}{r['threshold']:>14.6g}"
            f"   {'PASS' if r['pass'] else 'FAIL'}"
        )
    return "\n".join(out)
