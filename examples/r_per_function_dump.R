#!/usr/bin/env Rscript
# Per-function R output dump consumed by examples/function_by_function_R_parity.ipynb.
#
#   Rscript examples/r_per_function_dump.R data/fixture_spe_test.csv data/reference_output.json
#
# This is the same driver used for the parity gate -- every gated function
# already dumps its individual output there, so a second script would only
# duplicate it. Kept as a named entry point because NOTEBOOKS.md expects one.
#
# Fields consumed by Notebook 3, function by function:
#
#   readHoodData        -> cell_id
#   findNearCells       -> cells, distance
#   scanHoods (prox)    -> pm, tau_prox
#   scanHoods (smooth)  -> pm_smooth, tau_smooth, nll_smooth, nll_at_init, sub_idx
#   mergeByGroup        -> hoods            (discrete labels)
#   mergeByGroup        -> hoods_fuzzy, fuzzy_input   (continuousAnnotation = TRUE)
#   calcMetrics         -> entropy, perplexity
#   perplexityPermute   -> perplexity_p, n_perm
#   clustByHood         -> clusters, centroids, k_clust
#   plotColocal         -> colocal          (self_cor = TRUE)
#   plotColocal         -> mean_by_group    (self_cor = FALSE)

args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
driver <- normalizePath(file.path(here, "..", "tests", "r_reference_driver.R"))
cat("delegating to", driver, "\n")
system2("Rscript", c(driver, args))
