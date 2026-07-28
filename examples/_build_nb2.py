"""Build examples/tutorial_cosmx_lung.ipynb (Notebook 2)."""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# py-hoodscanR tutorial — CosMx lung cancer

A Python-only walkthrough of every public function, on real data: a 2661-cell
subset of a NanoString CosMx SMI non-small-cell lung cancer section
(`Lung9_Rep1`), the dataset the upstream hoodscanR package ships.

**What the method does.** For each cell it finds the *k* nearest cells,
converts their distances into a probability distribution with a
temperature-scaled softmax, and collapses that onto the cell-type labels. The
result is, per cell, a genuine probability vector over "neighbourhood types" —
so its **entropy** (and its exponential, **perplexity**) tells you how mixed
that cell's surroundings are.

Contents:

1. `load_spe_test` / `read_hood_data` — getting data in
2. `find_near_cells` — the k-nearest-cell search
3. `scan_hoods` — the distance-weighted softmax (and the bandwidth τ)
4. `merge_by_group` — collapsing onto cell types (hard and fuzzy labels)
5. `merge_hood_adata` — writing results back
6. `calc_metrics` — entropy and perplexity
7. `perplexity_permute` — is this neighbourhood unusually distinct?
8. `clust_by_hood` — neighbourhood-based clustering
9. `plot_tissue`, `plot_hood_mat`, `plot_prob_dist`, `plot_colocal`
10. The `HoodScanR` class API
11. Pitfalls
"""))

cells.append(code("""import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
import pyhoodscanr as ph

plt.rcParams.update({'figure.dpi': 110, 'font.size': 9})
print('py-hoodscanR', ph.__version__)"""))

cells.append(md("""## 1. Getting data in

`load_spe_test()` returns the bundled fixture. For your own data you need an
`AnnData` with coordinates in `.obsm['spatial']` and a cell-type column in
`.obs`; `read_hood_data` normalises it (it is the analogue of R's
`readHoodData`, which takes a `SpatialExperiment`)."""))

cells.append(code("""adata = ph.load_spe_test()
adata = ph.read_hood_data(adata, anno_col='cell_annotation')
print(adata)
adata.obs['cell_annotation'].value_counts()"""))

cells.append(code("""# From plain data frames instead (R's cell_pos_dat / cell_anno_dat path)
pos = pd.DataFrame({'cell_id': adata.obs_names,
                    'x': adata.obsm['spatial'][:, 0],
                    'y': adata.obsm['spatial'][:, 1]})
ann = pd.DataFrame({'cell_id': adata.obs_names,
                    'ct': adata.obs['cell_annotation'].to_numpy()})
ph.read_hood_data(cell_pos_dat=pos, cell_anno_dat=ann)"""))

cells.append(code("""ax = ph.plot_tissue(adata, color='cell_annotation', size=3, alpha=.9)
ax.set_title('CosMx Lung9_Rep1 — cell types')
plt.show()"""))

cells.append(md("""## 2. `find_near_cells` — the k-nearest-cell search

Returns two aligned frames: the neighbours' **cell types** and their
**distances**, both `n_cells x k`, sorted nearest-first.

| parameter | default | meaning |
|---|---|---|
| `k` | 100 | how many neighbours |
| `target_cell` | `None` | restrict the *query* set |
| `report_cell_id` | `False` | report ids instead of cell types |
| `report_dist` | `True` | also return distances |
| `anno_col` | `'cell_annotation'` | annotation column |
| `basis` | `'spatial'` | `obsm` key for coordinates |
| `tie_break` | `'stable'` | `(distance, index)` ordering — backend-independent |
"""))

cells.append(code("""fnc = ph.find_near_cells(adata, k=100)
print(fnc['cells'].iloc[:5, :5], '\\n')
print(fnc['distance'].iloc[:5, :5].round(2))"""))

cells.append(code("""d = fnc['distance'].to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(np.arange(1, 101), d.mean(0), lw=1.5)
ax[0].fill_between(np.arange(1, 101), np.percentile(d, 10, 0), np.percentile(d, 90, 0), alpha=.25)
ax[0].set_xlabel('neighbour rank'); ax[0].set_ylabel('distance (um)')
ax[0].set_title('how far is the k-th neighbour?')
ax[1].hist(d[:, 0], bins=60, color='steelblue')
ax[1].set_xlabel('distance to nearest cell (um)'); ax[1].set_title('nearest-neighbour distance')
fig.tight_layout(); plt.show()"""))

cells.append(md("""**Choosing k.** k sets the *maximum* spatial extent the method can see. The
softmax then down-weights the far end, so k is a ceiling rather than a hard
radius — this is why hoodscanR is less sensitive to k than a radius-based
counting method is to its radius. k=100 is the upstream default."""))

cells.append(md("""## 3. `scan_hoods` — the distance-weighted softmax

$$P_{ij} = \\frac{\\exp(-d_{ij}^2/\\tau)}{\\sum_{j'}\\exp(-d_{ij'}^2/\\tau)}$$

| parameter | default | meaning |
|---|---|---|
| `mode` | `'proximityFocused'` | `'proximityFocused'` fixes τ = median(d²)/5; `'smoothFadeout'` fits it |
| `tau` | `None` | override the bandwidth |
| `t_init` | `None` | starting value for the fit (default median(d²)) |
| `return_tau` | `False` | also return the τ actually used |
"""))

cells.append(code("""pm, tau = ph.scan_hoods(d, verbose=False, return_tau=True)
print(f'tau = {tau:.4f}   (= median(d^2)/5)')
print('rows sum to 1 :', np.allclose(pm.sum(1), 1))
print('shape         :', pm.shape)"""))

cells.append(code("""fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
for t, lab in [(tau/5, 'tau/5 (sharper)'), (tau, 'tau (default)'), (tau*5, '5*tau (flatter)')]:
    ax[0].plot(ph.soft_max_intl(d[:1], t)[0], lw=1.4, label=lab)
ax[0].axhline(1/100, color='k', ls=':', lw=1, label='uniform = counting methods')
ax[0].set_xlabel('neighbour rank'); ax[0].set_ylabel('probability')
ax[0].set_title('what tau controls (one cell)'); ax[0].legend(fontsize=7)

ent = ph.calculate_metrics(pm)['entropy']
ax[1].hist(ent, bins=60, color='indianred')
ax[1].set_xlabel('entropy over the 100 neighbours (bits)')
ax[1].set_title('effective neighbourhood size')
fig.tight_layout(); plt.show()"""))

cells.append(md("""### Fitting τ instead of fixing it

`mode='smoothFadeout'` maximises the likelihood of the soft assignment. It is
slower (it runs a BFGS optimisation) so it is usually run on a subsample."""))

cells.append(code("""pm_s, tau_s = ph.scan_hoods(d[:500], mode='smoothFadeout', verbose=False, return_tau=True)
print(f'heuristic tau (proximityFocused) = {ph.soft_max_intl.__module__ and tau:.2f}')
print(f'fitted    tau (smoothFadeout)    = {tau_s:.2f}')
print('\\nThe fitted tau is larger here, i.e. the likelihood prefers a flatter,')
print('longer-range neighbourhood than the median(d^2)/5 heuristic.')"""))

cells.append(md("""## 4. `merge_by_group` — collapse onto cell types

| parameter | default | meaning |
|---|---|---|
| `pm` | — | the `n x k` softmax matrix |
| `group_df` | — | `n x k` labels, **or** a `k x n_types` fuzzy matrix |
| `continuous_annotation` | `False` | treat `group_df` as fuzzy probabilities |
"""))

cells.append(code("""hoods = ph.merge_by_group(pm, fnc['cells'])
hoods.index = adata.obs_names
print('rows still sum to 1 :', np.allclose(hoods.sum(1), 1))
hoods.head().round(4)"""))

cells.append(md("""### Fuzzy / continuous annotations

If the labels come from deconvolution, feed the probabilities in directly
instead of taking an argmax first — uncertainty then propagates into the
neighbourhood profile. Neither `monkeybread` nor `squidpy` offers this."""))

cells.append(code("""rng = np.random.default_rng(0)
fuzzy = rng.random((100, 3)); fuzzy /= fuzzy.sum(1, keepdims=True)
fuzzy = pd.DataFrame(fuzzy, columns=['programme_A', 'programme_B', 'programme_C'])
hoods_fuzzy = ph.merge_by_group(pm, fuzzy, continuous_annotation=True)
print('shape:', hoods_fuzzy.shape, ' rows sum to 1:', np.allclose(hoods_fuzzy.sum(1), 1))
hoods_fuzzy.head().round(4)"""))

cells.append(md("""## 5–6. `merge_hood_adata` and `calc_metrics`

`calc_metrics` writes `entropy` and `perplexity` into `.obs`.

**Perplexity is the number to read.** It is the *effective number of
neighbourhood types* around the cell: 1 = a pure, distinct neighbourhood,
2 = a roughly 50/50 boundary, 6 = maximally mixed here."""))

cells.append(code("""ph.merge_hood_adata(adata, hoods)
ph.calc_metrics(adata, pm=hoods.to_numpy())
adata.obs[['entropy', 'perplexity']].describe().round(4)"""))

cells.append(code("""fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
ax[0].hist(adata.obs['perplexity'], bins=60, color='darkorange')
ax[0].set_xlabel('perplexity'); ax[0].set_title('most cells sit in a fairly distinct neighbourhood')
ph.plot_tissue(adata, color='perplexity', cmap='magma', size=3, ax=ax[1])
ax[1].set_title('perplexity in tissue space')
ph.plot_tissue(adata, color='entropy', cmap='viridis', size=3, ax=ax[2])
ax[2].set_title('entropy in tissue space')
fig.tight_layout(); plt.show()"""))

cells.append(code("""# perplexity per cell type
order = adata.obs.groupby('cell_annotation', observed=True)['perplexity'].median().sort_values().index
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.boxplot([adata.obs.loc[adata.obs['cell_annotation'] == c, 'perplexity'] for c in order],
           labels=list(order), showfliers=False)
ax.set_ylabel('perplexity'); ax.tick_params(axis='x', rotation=40)
ax.set_title('which cell types live in mixed neighbourhoods?')
fig.tight_layout(); plt.show()"""))

cells.append(md("""Tumour cells sit in the most homogeneous neighbourhoods (low perplexity —
tumour surrounded by tumour); dividing and immune cells are the most mixed.
That is the kind of statement this method exists to make."""))

cells.append(md("""## 7. `perplexity_permute` — is this neighbourhood unusually distinct?

Compares each cell's perplexity against the distribution obtained by randomly
re-assigning whole neighbourhood profiles across the tissue.

| parameter | default | meaning |
|---|---|---|
| `n_perm` | 1000 | permutations |
| `seed` | 42 | behaves exactly like R's `set.seed` |
| `exact` | `True` | use the row-permutation identity (exact, ~130x faster) |

A **small** p means unusually *distinct* (low perplexity)."""))

cells.append(code("""ph.perplexity_permute(adata, pm=hoods.to_numpy(), n_perm=1000, seed=42)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].scatter(adata.obs['perplexity'], adata.obs['perplexity_p'], s=3, alpha=.3)
ax[0].axhline(0.05, color='r', ls='--', lw=.8, label='p = 0.05'); ax[0].legend(fontsize=8)
ax[0].set_xlabel('perplexity'); ax[0].set_ylabel('permutation p')
ph.plot_tissue(adata, color='perplexity_p', cmap='viridis', size=3, ax=ax[1])
ax[1].set_title('p(perplexity)')
fig.tight_layout(); plt.show()
print(f"{(adata.obs['perplexity_p'] < 0.05).sum()} cells with p < 0.05 "
      f"({(adata.obs['perplexity_p'] < 0.05).mean():.1%})")"""))

cells.append(md("""## 8. `clust_by_hood` — neighbourhood-based clustering

k-means (**Hartigan–Wong**, R's default — ported here because scikit-learn only
ships Lloyd/Elkan) on the probability profiles.

| parameter | default | meaning |
|---|---|---|
| `k` | `0` → `2**n_types - 1` | number of clusters |
| `nstart` | 5 | random restarts |
| `iter_max` | 1000 | transfer sweeps |
| `seed` | 42 | matches R's `set.seed` |
"""))

cells.append(code("""ph.clust_by_hood(adata, pm_cols=list(hoods.columns), k=10, seed=42)
print(adata.obs['clusters'].value_counts().sort_index())
ax = ph.plot_tissue(adata, color='clusters', size=3)
ax.set_title('neighbourhood clusters (k=10)')
plt.show()"""))

cells.append(code("""cent = pd.DataFrame(adata.uns['hoodscanr']['centroids'], columns=hoods.columns,
                    index=[str(i+1) for i in range(10)])
fig, ax = plt.subplots(figsize=(6, 3.6))
im = ax.imshow(cent.to_numpy(), cmap='magma', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(6)); ax.set_xticklabels(cent.columns, rotation=60, ha='right', fontsize=7)
ax.set_yticks(range(10)); ax.set_yticklabels(cent.index, fontsize=7)
ax.set_ylabel('cluster'); ax.set_title('cluster centroids = neighbourhood signatures')
fig.colorbar(im, ax=ax); fig.tight_layout(); plt.show()
cent.round(3)"""))

cells.append(md("""## 9. Plotting"""))

cells.append(code("""ph.plot_hood_mat(hoods, n=10, seed=1)
plt.show()"""))

cells.append(code("""ph.plot_prob_dist(adata, pm_cols=list(hoods.columns), by_cluster=True,
                  plot_all=True, show_clusters=[str(i) for i in range(1, 9)])
plt.show()"""))

cells.append(code("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ph.plot_colocal(adata, ax=ax[0])
ph.plot_colocal(adata, self_cor=False, by_group='cell_annotation', cmap='magma', ax=ax[1])
fig.tight_layout(); plt.show()"""))

cells.append(md("""The left panel is the colocalisation analysis: endothelial and stromal
neighbourhoods co-occur, as do epithelial and dividing — reproducing the
observation in the upstream vignette."""))

cells.append(md("""## 10. The class API

Same computation, chainable, results accumulate in `hs.adata`."""))

cells.append(code("""hs = (ph.HoodScanR(ph.load_spe_test(), anno_col='cell_annotation')
        .find_near_cells(k=100)
        .scan_hoods()
        .merge_by_group()
        .calc_metrics()
        .clust_by_hood(k=10))
print(hs)
hs.adata.obs[['cell_annotation', 'entropy', 'perplexity', 'clusters']].head()"""))

cells.append(code("""# or in one call
hs2 = ph.HoodScanR(ph.load_spe_test(), anno_col='cell_annotation').run(k=100, n_clusters=10)
print(hs2)"""))

cells.append(md("""## 11. Pitfalls

1. **`.obs` gets one column per cell type.** `merge_hood_adata` writes the
   probability matrix into `.obs` (mirroring R's `colData`) *and* into
   `.obsm['hoods']`. If a cell type shares a name with an existing `.obs`
   column it will be overwritten. The column names are recorded in
   `adata.uns['hoodscanr']['pm_cols']`, so downstream calls do not need
   `pm_cols` spelled out.

2. **Perplexity is bounded by the number of cell types**, not by k. With 6 cell
   types it lies in [1, 6]. Comparing perplexity across datasets with different
   annotation granularity is meaningless.

3. **`k` is a ceiling, not a radius.** Increasing k mostly adds heavily
   down-weighted neighbours. If you want a hard spatial cut-off, use
   `squidpy`'s radius graph instead.

4. **`smoothFadeout` is not free.** It runs a BFGS fit; run it on a subsample
   and reuse the τ via `scan_hoods(d, tau=fitted_tau)`.

5. **Exact-distance ties at the k-th neighbour.** On a regular grid or with
   duplicated coordinates, several cells can be exactly equidistant at the
   boundary and which one is admitted is arbitrary. `tie_break='stable'`
   (the default) makes *this* package deterministic;
   `find_near_cells(..., warn_boundary_ties=True)` reports affected cells.

6. **Cluster labels are 1-based strings**, matching R's `clustByHood`.

7. **`numba` is optional but worth it** — `pip install pyhoodscanr[speed]`.
   Results are identical either way; the permutation test is ~130x slower
   without it.
"""))

cells.append(code("""fnc_warn = ph.find_near_cells(adata, k=100, warn_boundary_ties=True)
print('done — see the warning above if any boundary ties were found')"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbf.write(nb, 'tutorial_cosmx_lung.ipynb')
print('wrote tutorial_cosmx_lung.ipynb', len(cells), 'cells')
