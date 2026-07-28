# Discovery — py-hoodscanR

Protocol: [omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) **Step 0**.
Run date: 2026-07-28. Committed **before** any algorithmic code.

---

## 1. Direct check — is hoodscanR already ported?

```
$ python -m engine.discover_omicverse_deps --check hoodscanR
[discover] using cached repo list (95 repos)
## Discovery — `hoodscanR`

**No existing omicverse port found.** Safe to start a new port.
```

---

## 2. The real question — do Monkeybread or squidpy already do this?

`py-<pkg>` mirrors exist to fill *gaps*. Two Python packages advertise "cellular
niche / neighbourhood composition" and had to be ruled out before proceeding:

- **Monkeybread** 1.0.3 (Immunitas Therapeutics) — <https://doi.org/10.1101/2023.09.14.557736>
- **squidpy** 1.6.5 (Theis lab) — `sq.gr.calculate_niche`, `sq.gr.nhood_enrichment`, `sq.gr.co_occurrence`

Both were installed/unpacked and their **source read**, not just their docs.

### 2.1 What hoodscanR actually computes (from `hoodscanR-ref/R/`)

| R function | File | What it computes |
|---|---|---|
| `readHoodData` | `read_data.R` | Normalises a `SpatialExperiment` → coords + `cell_annotation` column |
| `findNearCells` | `find_near_cells.R` | k-NN via `RANN::nn2(searchtype="priority")`, `k+1` then drop self. Returns **both** a cell-type matrix (cells × k) and a **distance** matrix (cells × k) |
| `scanHoods` | `soft_neighbourhood.R` | **The core.** Distance-weighted softmax: `P_ij = exp(-d_ij²/τ) / Σ_j exp(-d_ij²/τ)`. Two modes: `proximityFocused` (τ = median(d²)/5) and `smoothFadeout` (τ fitted by BFGS minimising `-Σ log(P+1e-8)`) |
| `mergeByGroup` | `merge_by_group.R` | Collapses the cells × k soft weights onto cells × celltypes by summing weights within each label. Also supports **fuzzy/continuous** annotations (`pm %*% A`, renormalised) |
| `calcMetrics` + `calculate_metrics` (Rcpp) | `calc_metrics.R`, `src/cal_metrics.cpp` | Per-cell Shannon **entropy** (base 2) of the probability vector and **perplexity** = 2^H |
| `perplexityPermute` | `calc_metrics.R` | Per-cell permutation-test **p-value** for perplexity (n_perm=1000, row-shuffle) |
| `clustByHood` | `pm_clust.R` | k-means (**Hartigan–Wong**, `nstart=5`, default `k = 2^ncol − 1`) on the probability matrix |
| `mergeHoodSpe` | `merge_pm_spe.R` | Writes the probability matrix back into `colData` |
| `plotColocal` | `plot_heatmap.R` | Pearson correlation **between neighbourhood columns** → colocalisation heatmap |
| `plotHoodMat`, `plotProbDist`, `plotTissue` | `plot_pm.R`, `plot_pd.R`, `plot_tissue.R` | Visualisation |

The distinguishing idea in one line: **every neighbour contributes a
*continuous, distance-decaying* weight, and the per-cell result is a genuine
probability distribution whose *entropy* is the reported quantity.**

### 2.2 What Monkeybread computes

`monkeybread/calc/_neighborhood_profile.py::neighborhood_profile` —
verbatim from the wheel:

```python
cell_to_neighbor_counts = {
    cell: Counter(cell_to_group[c] for c in cell_to_neighbors[cell]) ...
}
neighbors_df = pd.DataFrame(cell_to_neighbor_counts).T.fillna(0)
if normalize_counts:
    neighbors_df = neighbors_df.apply(lambda arr: arr / np.sum(arr), axis=1)
```

A `Counter`. Every neighbour inside the radius (or inside the k-NN set) counts
exactly **1.0**, regardless of whether it is touching the cell or sitting at the
edge of the radius. The result is then **z-scored** (`sc.pp.scale`) and clipped —
after which the row is no longer a probability vector at all, so entropy is
undefined on it. `cellular_niches()` then Leiden-clusters those z-scores.

Monkeybread's other modules (`cell_neighbors`, `cell_density`,
`shortest_distances`, `number_neighbors`, `ligand_receptor`) are a different
question set — contact/proximity statistics between named cell-type pairs, not a
per-cell composition distribution.

### 2.3 What squidpy computes

`squidpy/gr/_niche.py::_calculate_neighborhood_profile` — verbatim:

```python
abs_freq = np.zeros((m, len(unique_categories)), dtype=int)
np.add.at(abs_freq, (np.arange(m)[:, None], cat_values), 1)
rel_freq = abs_freq / k
```

Again integer counts divided by `k`. `flavor="utag"` is `A @ X` (adjacency times
expression); `flavor="cellcharter"` is n-hop aggregation + GMM. `nhood_enrichment`
is a **cluster-pair z-score by permutation** — a tissue-level summary, not a
per-cell vector. `co_occurrence` is a ratio of conditional probabilities over
distance bins, again per cell-type-pair, not per cell.

### 2.4 Overlap matrix

| Capability | hoodscanR | Monkeybread | squidpy |
|---|---|---|---|
| k-NN / radius neighbour graph | ✅ | ✅ | ✅ |
| Per-cell composition over cell types | ✅ | ✅ | ✅ |
| **Distance-weighted (soft) neighbour contribution** | ✅ softmax on `d²` | ❌ hard `Counter` | ❌ hard `abs_freq/k` |
| **Bandwidth τ as a fitted parameter** (`smoothFadeout`, BFGS on NLL) | ✅ | ❌ | ❌ |
| **Row is a true probability distribution (Σ=1)** | ✅ | ✗ (z-scored/clipped) | ✗ (z-scored) |
| **Per-cell neighbourhood entropy (base-2)** | ✅ | ❌ | ❌ |
| **Per-cell perplexity (2^H, "effective # of neighbourhoods")** | ✅ | ❌ | ❌ |
| **Permutation p-value on per-cell perplexity** | ✅ | ❌ | ❌ (squidpy permutes at the *cluster-pair* level, not per cell) |
| Fuzzy / continuous cell-type annotation input | ✅ | ❌ | ❌ |
| Neighbourhood colocalisation (corr between hood columns) | ✅ | ❌ | ~ (`nhood_enrichment`, different statistic) |
| Clustering of the profiles into niches | ✅ k-means Hartigan–Wong | ✅ Leiden | ✅ Leiden / GMM |

### 2.5 Decision — **PORT**

Monkeybread and squidpy both stop at *"which cell types are near me, counted"*.
hoodscanR answers *"what is the probability distribution of my neighbourhood, and
how uncertain is it"*. Concretely, the capability with **no Python equivalent** is:

1. **Distance-weighted soft neighbourhood probability** — `exp(-d²/τ)` softmax
   over the k-NN distances, so a neighbour 5 µm away and one 200 µm away do not
   both count as 1. Neither Python package weights by distance at all.
2. **A fitted bandwidth τ** (`smoothFadeout`) rather than a user-chosen radius.
3. **Per-cell entropy / perplexity** of that distribution — the measurement you
   want *before* declaring anything a niche. This is only definable because (1)
   leaves the row summing to 1; both Python packages destroy that property by
   z-scoring.
4. **Per-cell permutation p-value** on perplexity.
5. **Fuzzy annotation support** (`continuousAnnotation=TRUE`) — propagate
   deconvolution/label-uncertainty into the neighbourhood profile.

Items 1–5 are the whole scientific claim of the hoodscanR preprint
(<https://doi.org/10.1101/2024.03.26.586902>) and none of them exists in Python.
Proceeding to Step 1.

**Scoped out** (already covered, do not reimplement): Leiden-based niche
discovery — users who want that should use `sq.gr.calculate_niche` or
`mb.calc.cellular_niches`. This port mirrors hoodscanR's own k-means path only,
and documents the alternative.

---

## 3. Dependency audit

`hoodscanR-ref/DESCRIPTION` `Imports:` field, mapped against `github.com/omicverse`:

| R dep | omicverse port | Decision |
|---|---|---|
| `RANN` (ANN k-NN, `nn2`) | — | native Python equivalent: `sklearn.neighbors.NearestNeighbors` (exact mode; RANN default `eps=0` is also exact) |
| `Rcpp` (`src/cal_metrics.cpp`) | — | native: vectorised NumPy; the C++ kernel is 20 lines of entropy |
| `SpatialExperiment` / `SummarizedExperiment` | — | native: `anndata.AnnData` (`.obsm['spatial']`, `.obs`) |
| `stats::optim(method="BFGS")` | — | **no drop-in equivalent** — `scipy.optimize.minimize(method="BFGS")` is a *different* implementation (different finite-difference step, different line search). R's `vmmin` + `ndeps=1e-3` central difference are ported directly (see `pyhoodscanr/_roptim.py`) |
| `stats::kmeans(algorithm="Hartigan-Wong")` | — | **no drop-in equivalent** — scikit-learn only ships Lloyd/Elkan. AS-136 Hartigan–Wong ported directly (see `pyhoodscanr/_kmeans_hw.py`) |
| `ggplot2`, `ComplexHeatmap`, `circlize`, `scico`, `grid` | — | native: `matplotlib` / `seaborn` |
| `knitr`, `rmarkdown`, `rlang`, `methods`, `utils` | — | out of scope (R plumbing) |

**No omicverse mirror reused** — hoodscanR's dependency set is base-R numerics
plus plotting; there is nothing in the ecosystem to inherit. Two R *behaviours*
(`vmmin`, Hartigan–Wong) had to be ported by hand because the obvious Python
substitutes are different algorithms, and substituting them would have silently
broken parity.
