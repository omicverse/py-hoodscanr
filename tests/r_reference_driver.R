#!/usr/bin/env Rscript
# Reference runner for py-hoodscanR.
#
#   Rscript tests/r_reference_driver.R <fixture.csv> <output.json>
#
# Runs the upstream hoodscanR pipeline (exactly the sequence in the package
# vignette) on the canonical fixture and dumps every gated output to JSON.
# This file is the EXECUTABLE SPEC — the Python port is diffed against it.

suppressPackageStartupMessages({
  library(hoodscanR)
  library(SpatialExperiment)
  library(SummarizedExperiment)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
fixture_path <- args[1]
output_path <- args[2]
K <- if (length(args) >= 3) as.integer(args[3]) else 100L
N_PERM <- if (length(args) >= 4) as.integer(args[4]) else 1000L
K_CLUST <- if (length(args) >= 5) as.integer(args[5]) else 10L

fx <- read.csv(fixture_path, stringsAsFactors = FALSE, colClasses = "character")
cell_id <- fx$cell_id
pos <- data.frame(cell_id = cell_id, x = as.numeric(fx$x), y = as.numeric(fx$y),
                  stringsAsFactors = FALSE)
ann <- data.frame(cell_id = cell_id, cell_annotation = fx$cell_annotation,
                  stringsAsFactors = FALSE)

spe <- readHoodData(cell_pos_dat = pos, cell_anno_dat = ann)

## ---- 1. k nearest cells -------------------------------------------------
t0 <- Sys.time()
fnc <- findNearCells(spe, k = K)
t_fnc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

## ---- 2. soft neighbourhood, proximityFocused ---------------------------
t0 <- Sys.time()
pm <- scanHoods(fnc$distance)
t_scan <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
tau_prox <- median(fnc$distance**2) / 5

## ---- 2b. soft neighbourhood, smoothFadeout (BFGS-fitted tau) -----------
# Fit tau on a deterministic subsample so the BFGS trace is cheap but exact.
sub_idx <- seq_len(min(500L, nrow(fnc$distance)))
m_sub <- fnc$distance[sub_idx, , drop = FALSE]
t0 <- Sys.time()
pm_smooth <- scanHoods(m_sub, mode = "smoothFadeout")
t_smooth <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
# recover the fitted tau by re-running the same optimisation
f_nll_local <- function(m, t) {
  mm <- m**2
  mm <- -mm / t
  e <- exp(mm)
  p <- sweep(e, 1, rowSums(e), "/")
  -sum(log(p + 1e-8))
}
opt <- stats::optim(par = median(m_sub**2), fn = f_nll_local, m = m_sub, method = "BFGS")
tau_smooth <- opt$par
nll_smooth <- opt$value
nll_at_init <- f_nll_local(m_sub, median(m_sub**2))

## ---- 3. merge by group --------------------------------------------------
t0 <- Sys.time()
hoods <- mergeByGroup(pm, fnc$cells)
t_merge <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

## ---- 3b. merge by group, fuzzy / continuous annotation -----------------
set.seed(42)
n_types <- 3L
fuzzy <- matrix(stats::runif(ncol(pm) * n_types), ncol = n_types)
fuzzy <- sweep(fuzzy, 1, rowSums(fuzzy), "/")
colnames(fuzzy) <- paste0("celltype_", seq(n_types))
hoods_fuzzy <- mergeByGroup(pm, fuzzy, continuousAnnotation = TRUE)

## ---- 4. entropy / perplexity -------------------------------------------
spe <- mergeHoodSpe(spe, hoods)
t0 <- Sys.time()
spe <- calcMetrics(spe, pm_cols = colnames(hoods))
t_metrics <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cd <- as.data.frame(colData(spe), optional = TRUE)

## ---- 5. permutation p-value on perplexity ------------------------------
set.seed(42)
t0 <- Sys.time()
spe <- perplexityPermute(spe, pm_cols = colnames(hoods), n_perm = N_PERM)
t_perm <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("\n")
cd2 <- as.data.frame(colData(spe), optional = TRUE)

## ---- 6. clustering ------------------------------------------------------
set.seed(42)
t0 <- Sys.time()
spe <- clustByHood(spe, pm_cols = colnames(hoods), k = K_CLUST)
t_clust <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cd3 <- as.data.frame(colData(spe), optional = TRUE)

## ---- 7. colocalisation matrix ------------------------------------------
colocal <- stats::cor(as.matrix(cd3[, colnames(hoods)]))

## ---- 8. mean probability by group (plotColocal self_cor = FALSE) -------
agg <- stats::aggregate(cd3[, colnames(hoods)],
                        by = list(group = cd3[["cell_annotation"]]),
                        FUN = mean)
mean_by_group <- as.matrix(agg[, -1])
rownames(mean_by_group) <- agg$group

out <- list(
  cell_id = cell_id,
  celltypes = colnames(hoods),
  k = K,
  distance = unname(as.matrix(fnc$distance)),
  cells = unname(as.matrix(fnc$cells)),
  tau_prox = tau_prox,
  pm = unname(as.matrix(pm)),
  hoods = unname(as.matrix(hoods)),
  hoods_fuzzy = unname(as.matrix(hoods_fuzzy)),
  fuzzy_input = unname(as.matrix(fuzzy)),
  sub_idx = sub_idx,
  tau_smooth = tau_smooth,
  nll_smooth = nll_smooth,
  nll_at_init = nll_at_init,
  tau_smooth_init = median(m_sub**2),
  pm_smooth = unname(as.matrix(pm_smooth)),
  entropy = as.numeric(cd$entropy),
  perplexity = as.numeric(cd$perplexity),
  perplexity_p = as.numeric(cd2$perplexity_p),
  n_perm = N_PERM,
  clusters = as.integer(cd3$clusters),
  k_clust = K_CLUST,
  centroids = unname(as.matrix(metadata(spe)$centroids)),
  colocal = unname(colocal),
  mean_by_group = unname(mean_by_group),
  mean_by_group_rows = rownames(mean_by_group),
  timings = list(find_near_cells = t_fnc, scan_hoods = t_scan,
                 scan_hoods_smooth = t_smooth,
                 merge_by_group = t_merge, calc_metrics = t_metrics,
                 perplexity_permute = t_perm, clust_by_hood = t_clust)
)

# --- serialisation -------------------------------------------------------
# jsonlite truncates doubles to ~15 significant digits even with digits = NA,
# which would inject ~1e-12 of fake error into an element-wise 1e-8 gate.
# Numeric arrays therefore go out as raw little-endian float64 alongside the
# JSON, which round-trips exactly.
bindir <- paste0(tools::file_path_sans_ext(output_path), "_bin")
dir.create(bindir, showWarnings = FALSE, recursive = TRUE)

numeric_fields <- c("distance", "pm", "hoods", "hoods_fuzzy", "fuzzy_input",
                    "pm_smooth", "entropy", "perplexity", "perplexity_p",
                    "centroids", "colocal", "mean_by_group")
shapes <- list()
for (nm in numeric_fields) {
  v <- out[[nm]]
  dm <- if (is.null(dim(v))) c(length(v), 1L) else dim(v)
  # write row-major so numpy can reshape directly
  vv <- if (is.null(dim(v))) as.numeric(v) else as.numeric(t(v))
  con <- file(file.path(bindir, paste0(nm, ".f64")), "wb")
  writeBin(vv, con, size = 8, endian = "little")
  close(con)
  shapes[[nm]] <- dm
  out[[nm]] <- NULL
}
out$`_bin_shapes` <- shapes
out$`_bin_dir` <- basename(bindir)

jsonlite::write_json(out, output_path, auto_unbox = TRUE, digits = NA, matrix = "rowmajor")
cat("wrote", output_path, "and", bindir, "\n")
