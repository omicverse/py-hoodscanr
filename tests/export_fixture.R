#!/usr/bin/env Rscript
# Export the canonical fixture (hoodscanR::spe_test) to language-neutral files
# so the Python port and the R reference read *bit-identical* input.
#
# Usage: Rscript tests/export_fixture.R <outdir>

suppressPackageStartupMessages({
  library(hoodscanR)
  library(SpatialExperiment)
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "data"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

data("spe_test", package = "hoodscanR")
spe <- readHoodData(spe, anno_col = "celltypes")

coords <- SpatialExperiment::spatialCoords(spe)
colnames(coords) <- c("x", "y")
anno <- as.character(SummarizedExperiment::colData(spe)[, "cell_annotation"])

df <- data.frame(
  cell_id = colnames(spe),
  x = coords[, "x"],
  y = coords[, "y"],
  cell_annotation = anno,
  stringsAsFactors = FALSE
)

# 22 significant digits => exact f64 round-trip
write.csv(format(df, digits = 22, trim = TRUE), file.path(outdir, "fixture_spe_test.csv"),
          row.names = FALSE, quote = TRUE)

cat("wrote", file.path(outdir, "fixture_spe_test.csv"), nrow(df), "cells\n")
