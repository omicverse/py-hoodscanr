## R function coverage audit

### Coverage summary

| Category | Ported | Total | % |
|---|---|---|---|
| Exported R functions | 12 | 12 | 100.0% |
| Internal helpers (reachable) | 2 | 5 | 40.0% |

_Python package exposes 50 unique names._

### Exported R functions

| R function | Python equivalent | Status |
|---|---|---|
| `calcMetrics` | `calc_metrics` | ✅ ported |
| `clustByHood` | `clust_by_hood` | ✅ ported |
| `findNearCells` | `find_near_cells` | ✅ ported |
| `mergeByGroup` | `merge_by_group` | ✅ ported |
| `mergeHoodSpe` | `merge_hood_spe` | ✅ ported |
| `perplexityPermute` | `perplexity_permute` | ✅ ported |
| `plotColocal` | `plot_colocal` | ✅ ported |
| `plotHoodMat` | `plot_hood_mat` | ✅ ported |
| `plotProbDist` | `plot_prob_dist` | ✅ ported |
| `plotTissue` | `plot_tissue` | ✅ ported |
| `readHoodData` | `read_hood_data` | ✅ ported |
| `scanHoods` | `scan_hoods` | ✅ ported |

### Internal helpers reachable from exports

| R helper | File | Python equivalent | Status |
|---|---|---|---|
| `calculate_metrics` | `RcppExports.R` | `calculate_metrics` | ✅ ported |
| `col2rownames` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `rownames2col` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `soft_max_intl` | `soft_neighbourhood.R` | `soft_max_intl` | ✅ ported |
| `tissue_theme` | `plot_tissue.R` | `—` | 🔸 missing-or-inlined |

