# py-hoodscanR

A pure-Python re-implementation of the Bioconductor package
[**hoodscanR**](https://github.com/DavisLaboratory/hoodscanR) — spatial
cellular *neighbourhood* scanning for single-cell-resolution spatial
transcriptomics.

For every cell it computes a **distance-weighted probability distribution over
the cell types around it**, and then the **entropy** and **perplexity** of that
distribution — a per-cell measure of how mixed the local tissue is. Validated
against the R package output on the upstream package's own CosMx fixture; see
[the parity table](#r-parity).

```
pip install pyhoodscanr
```

---

## Why this exists (and when to use something else)

Python already has good tools for *counting* what is near a cell —
[`squidpy`](https://squidpy.readthedocs.io) (`sq.gr.calculate_niche`,
`nhood_enrichment`, `co_occurrence`) and
[`monkeybread`](https://monkeybread.readthedocs.io)
(`mb.calc.neighborhood_profile`, `cellular_niches`). Both build a per-cell
composition vector by counting neighbours inside a radius or a k-NN set,
z-scoring the counts, and clustering the result.

hoodscanR asks a different question, and four of its answers have no Python
equivalent:

| | hoodscanR | monkeybread | squidpy |
|---|---|---|---|
| Per-cell composition over cell types | ✅ | ✅ | ✅ |
| **Distance-weighted** neighbour contribution | ✅ `exp(−d²/τ)` | ❌ `Counter` | ❌ `abs_freq / k` |
| **Fitted** bandwidth τ (max-likelihood) | ✅ | ❌ | ❌ |
| Row is a **true probability distribution** | ✅ | ✗ z-scored | ✗ z-scored |
| **Per-cell neighbourhood entropy / perplexity** | ✅ | ❌ | ❌ |
| **Per-cell permutation p-value** on perplexity | ✅ | ❌ | ❌ (squidpy permutes cluster *pairs*) |
| Fuzzy / continuous cell-type input | ✅ | ❌ | ❌ |

The distance weighting is the load-bearing difference. In a counting method a
neighbour 5 µm away and one at the edge of the radius both contribute exactly
1. Here the contribution decays smoothly, the row still sums to 1, and *because*
it sums to 1 the entropy is well defined — which is what lets you ask "how
mixed is this cell's neighbourhood?" **before** committing to a niche label.

**Use `squidpy` or `monkeybread` instead** if you want Leiden-based niche
*discovery* — that is well covered there and deliberately not duplicated here.

---

## Quick start

```python
import pyhoodscanr as ph

adata = ph.load_spe_test()          # bundled CosMx fixture (2661 cells, 6 types)

hs = (ph.HoodScanR(adata, anno_col="cell_annotation")
        .find_near_cells(k=100)     # k nearest cells + their distances
        .scan_hoods()               # distance-weighted softmax  ->  P (n x k)
        .merge_by_group()           # collapse onto cell types   ->  H (n x 6)
        .calc_metrics()             # entropy + perplexity per cell
        .perplexity_permute(1000)   # per-cell p-value
        .clust_by_hood(k=10))       # k-means on the profiles

hs.adata.obs[["entropy", "perplexity", "perplexity_p", "clusters"]].head()
```

Reading the output: **perplexity is the effective number of neighbourhood types
around a cell.** ~1 means a pure, distinct neighbourhood; ~2 means the cell sits
on a roughly 50/50 boundary; higher means a well-mixed region.

```python
import matplotlib.pyplot as plt
ph.plot_tissue(hs.adata, color="perplexity", cmap="magma", size=3)
ph.plot_colocal(hs.adata)          # which neighbourhoods co-occur
plt.show()
```

### Functional API (one-to-one with R)

```python
adata = ph.read_hood_data(adata, anno_col="celltypes")
fnc   = ph.find_near_cells(adata, k=100)          # {"cells", "distance"}
pm    = ph.scan_hoods(fnc["distance"].to_numpy())
hoods = ph.merge_by_group(pm, fnc["cells"])
ph.merge_hood_spe(adata, hoods)
ph.calc_metrics(adata, pm=hoods.to_numpy())
ph.clust_by_hood(adata, pm_cols=list(hoods.columns), k=10)
```

### Fitting the bandwidth instead of fixing it

```python
pm, tau = ph.scan_hoods(dist, mode="smoothFadeout", return_tau=True)
```

`proximityFocused` (default) sets `τ = median(d²)/5`. `smoothFadeout` fits `τ`
by maximum likelihood using a faithful port of R's `optim(method="BFGS")`.

### Fuzzy cell-type annotations

If your labels come from deconvolution rather than hard assignment, feed the
probabilities straight in instead of taking an argmax first:

```python
hoods = ph.merge_by_group(pm, fuzzy_k_by_type, continuous_annotation=True)
```

---

## What's included

| R function | Python | Notes |
|---|---|---|
| `readHoodData` | `read_hood_data` | `SpatialExperiment` → `AnnData` |
| `findNearCells` | `find_near_cells` | + `tie_break` for backend-independent output |
| `scanHoods` | `scan_hoods` | both `proximityFocused` and `smoothFadeout` |
| `mergeByGroup` | `merge_by_group` | incl. `continuous_annotation` |
| `mergeHoodSpe` | `merge_hood_spe` / `merge_hood_adata` | |
| `calcMetrics` | `calc_metrics`, `calculate_metrics` | |
| `perplexityPermute` | `perplexity_permute` | |
| `clustByHood` | `clust_by_hood` | Hartigan–Wong k-means |
| `plotTissue` | `plot_tissue` | matplotlib |
| `plotHoodMat` | `plot_hood_mat` | matplotlib |
| `plotProbDist` | `plot_prob_dist` | matplotlib |
| `plotColocal` | `plot_colocal` | matplotlib; `return_matrix=True` for the numbers |

12/12 exported R functions ported (`data/r_function_audit.md`).

Three R behaviours had to be ported by hand because the obvious Python
substitutes are *different algorithms*, and swapping them in would have
silently changed results:

- **`pyhoodscanr/_rrng.py`** — R's Mersenne-Twister, `set.seed` scrambling and
  R ≥ 3.6 rejection-sampling `sample.int`. NumPy's MT19937 is the same core but
  a different stream. Bit-exact against R.
- **`pyhoodscanr/_roptim.py`** — R's `vmmin` BFGS with `ndeps = 1e-3` absolute
  central differences. SciPy's BFGS uses a different step and line search and
  lands on a different τ. Bit-exact against R (par, value *and* call counts).
- **`pyhoodscanr/_kmeans_hw.py`** — Hartigan–Wong (AS 136). scikit-learn ships
  only Lloyd/Elkan, which is a strictly weaker local search.

---

## R parity

Gate pre-registered in [`data/manifest.yaml`](data/manifest.yaml) **before** any
Python was written, and read-only afterwards. Fixture: `hoodscanR::spe_test`
(NanoString CosMx SMI, Lung9_Rep1; 2661 cells, 6 cell types), k = 100,
n_perm = 1000, k_clust = 10. Reference: hoodscanR 1.7.2 on R 4.4.3.

| Output | Metric | Threshold | Measured | |
|---|---|---|---|---|
| k-NN distances | max abs err | ≤ 1e-8 | **5.68e-13** | ✅ |
| soft probability matrix `P` (n×100) | max abs err | ≤ 1e-8 | **5.00e-16** | ✅ |
| merged probabilities `H` (n×6) | mean per-cell cosine | ≥ 0.9999 | **0.999999999998** | ✅ |
| merged probabilities `H` | max per-cell total variation | ≤ 1e-3 | **4.27e-5** | ✅ |
| entropy | Pearson r | ≥ 0.99 | **1.000000000000** | ✅ |
| perplexity | Pearson r | ≥ 0.99 | **1.000000000000** | ✅ |
| fitted τ (`smoothFadeout`) | relative error | ≤ 1e-3 | **1.12e-15** | ✅ |
| perplexity permutation p | Pearson r | ≥ 0.99 | **1.0** (element-wise identical) | ✅ |
| clusters | ARI | ≥ 0.95 | **1.0** (identical labels) | ✅ |
| colocalisation matrix | max abs err | ≤ 1e-8 | 8.48e-8 | ⚠️ |

**9 of 10 pass.** The one that does not is fully diagnosed: 2 of 2661 cells
(0.075%) have two candidate neighbours at *exactly* the same f64 distance
competing for the 100th slot, and R's ANN and scikit-learn admit different
ones. Remove those two cells and the correlation matrix agrees with R to
**2.2e-16**. The threshold was not widened — see
[`MATH.md`](MATH.md) §4 and [`RECONSTRUCTION_REPORT.md`](RECONSTRUCTION_REPORT.md) §4.

### Reproduce it yourself

```bash
Rscript tests/export_fixture.R data
Rscript tests/r_reference_driver.R data/fixture_spe_test.csv data/reference_output.json
python tests/run_candidate.py data/fixture_spe_test.csv data/candidate_output.npz
pytest tests/ -q
```

The R reference output is committed, so `pytest` alone works without R.

---

## Performance

Full pipeline on the canonical fixture, warm-up discarded, mean ± sd of 5 runs
(Sherlock `sh04-04n05`; `python tests/benchmark.py 5`):

| stage | hoodscanR (R) | py-hoodscanR | speedup |
|---|---|---|---|
| `findNearCells` | 0.4792 ± 0.004 s | 0.0420 ± 0.000 s | 11.4× |
| `scanHoods` | 0.0157 ± 0.000 s | 0.0034 ± 0.000 s | 4.6× |
| `scanHoods` (smoothFadeout) | 0.0359 ± 0.000 s | 0.0086 ± 0.000 s | 4.2× |
| `mergeByGroup` | 0.0251 ± 0.000 s | 0.0167 ± 0.000 s | 1.5× |
| `calcMetrics` | 0.0380 ± 0.001 s | 0.0003 ± 0.000 s | 109× |
| `perplexityPermute` (1000 perms) | 0.9313 ± 0.007 s | 0.1152 ± 0.000 s | 8.1× |
| `clustByHood` (k=10, nstart=5) | 0.0548 ± 0.001 s | 0.0388 ± 0.000 s | 1.4× |
| **total** | **1.5800 s** | **0.2251 s** | **7.0×** |

Every accepted optimisation is an **exact** identity — accuracy is flat to the
last recorded digit across all of them ([`ITERATION_LOG.md`](ITERATION_LOG.md),
[`examples/evolution.png`](examples/evolution.png)). `numba` is optional
(`pip install pyhoodscanr[speed]`); without it the results are identical and
the permutation test is ~130× slower.

---

## Notebooks

- [`examples/compare_R_vs_Python.ipynb`](examples/compare_R_vs_Python.ipynb) — pipeline-level parity, one figure per gated output
- [`examples/tutorial_cosmx_lung.ipynb`](examples/tutorial_cosmx_lung.ipynb) — Python-only walkthrough of every public function
- [`examples/function_by_function_R_parity.ipynb`](examples/function_by_function_R_parity.ipynb) — R⇄Python parameter dictionary with per-function numerical comparison
- [`examples/evolution.ipynb`](examples/evolution.ipynb) — the acceleration trajectory, iteration by iteration

All pre-executed with outputs committed.

---

## Relationship to omicverse

Built with the [omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr)
protocol (Discovery → template → dual environments → Equivalence/Acceleration
loop → pre-registered gate → release). `pyhoodscanr` has no omicverse runtime
dependency and is usable standalone.

## Citation

Cite the original method:

> Liu N, Bhuva DD, Mohamed A, Bokelund M, Kulasinghe A, Tan CW, Davis MJ.
> *hoodscanR: profiling single-cell neighborhoods in spatial transcriptomics
> data.* bioRxiv (2024). https://doi.org/10.1101/2024.03.26.586902

and, if the Python port itself mattered to your work, this repository.

## License

GPL-3.0-or-later, matching upstream hoodscanR (GPL-3 + file LICENSE).
