#!/usr/bin/env Rscript
# XCMS feature extraction (reference peak-picker baseline, DEFAULT parameters).
# Output: <MINUT_FEATURES_DIR>/<dataset>_xcms.csv  (samples x features + label)
#
# Usage:  Rscript extract_xcms.R <dataset> <manifest.csv>
#
# Environment-driven paths:
#   MINUT_FEATURES_DIR   output dir (default: outputs/features)
#   MINUT_DATA_DIR       root that holds the raw files (default: data)
# Manifest columns: sample_name, file_path, label. `file_path` may be absolute
# or just a file name resolved under MINUT_DATA_DIR.

suppressPackageStartupMessages({
  library(xcms)
  library(MsExperiment)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: extract_xcms.R <dataset> <manifest.csv>")
dataset  <- args[[1]]
manifest <- args[[2]]

out_dir  <- Sys.getenv("MINUT_FEATURES_DIR", unset = "outputs/features")
data_dir <- Sys.getenv("MINUT_DATA_DIR", unset = "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

man <- read.csv(manifest, stringsAsFactors = FALSE)

# Resolve each raw file: use file_path if it exists, else data_dir/<basename>.
resolve_raw <- function(fp) {
  if (file.exists(fp)) return(fp)
  cand <- file.path(data_dir, basename(fp))
  if (file.exists(cand)) cand else file.path(data_dir, fp)
}
man$resolved <- vapply(man$file_path, resolve_raw, character(1))
cat(sprintf("dataset=%s  n_samples=%d\n", dataset, nrow(man)))

register(SerialParam())

sd <- data.frame(sample = man$sample_name, label = man$label, stringsAsFactors = FALSE)
mse <- readMsExperiment(spectraFiles = man$resolved, sampleData = sd)
cat(sprintf("loaded %d files\n", length(mse)))

cwp <- CentWaveParam()  # defaults: ppm=25, peakwidth=c(20,50), snthresh=10, ...
t0 <- proc.time()[["elapsed"]]
res <- findChromPeaks(mse, param = cwp)
cat(sprintf("findChromPeaks done in %.1f s, n_peaks=%d\n",
            proc.time()[["elapsed"]] - t0, nrow(chromPeaks(res))))

pdp <- PeakDensityParam(sampleGroups = rep(1L, nrow(man)), bw = 30, minFraction = 0.25)
res <- groupChromPeaks(res, param = pdp)
cat(sprintf("groupChromPeaks: n_features=%d\n", nrow(featureDefinitions(res))))

res <- tryCatch(fillChromPeaks(res, param = ChromPeakAreaParam()),
                error = function(e) { cat("fillChromPeaks skipped:", conditionMessage(e), "\n"); res })

fv <- featureValues(res, value = "into", method = "medret")
M <- t(fv)
M[is.na(M)] <- 0
colnames(M) <- paste0("feat_", seq_len(ncol(M)))
# featureValues columns follow the sampleData (manifest) order; map row names
# (raw-file basenames) back to the manifest sample_name/label robustly.
row_keys <- rownames(M)
man_keys <- basename(man$resolved)
idx <- match(row_keys, man_keys)
if (anyNA(idx)) idx <- seq_len(nrow(M))  # fall back to positional order
samples <- man$sample_name[idx]
labels  <- man$label[idx]
df <- data.frame(sample_name = samples, M,
                 label = labels,
                 check.names = FALSE, stringsAsFactors = FALSE)
out <- file.path(out_dir, paste0(dataset, "_xcms.csv"))
write.csv(df, out, row.names = FALSE)
cat(sprintf("-> %d samples x %d features -> %s\n", nrow(df), ncol(df) - 2, out))
cat(sprintf("Total elapsed: %.2f min\n", (proc.time()[["elapsed"]] - t0) / 60))

# Record the exact software versions next to the feature table (R, xcms, MsExperiment, BiocParallel, ...).
cat(sprintf("xcms %s | MsExperiment %s | BiocParallel %s | %s\n",
            as.character(packageVersion("xcms")), as.character(packageVersion("MsExperiment")),
            as.character(packageVersion("BiocParallel")), R.version.string))
session_file <- file.path(out_dir, paste0(dataset, "_xcms_sessionInfo.txt"))
writeLines(c(capture.output(print(sessionInfo())), "",
             sprintf("CentWaveParam: ppm=%s peakwidth=%s snthresh=%s prefilter=%s mzdiff=%s",
                     cwp@ppm, paste(cwp@peakwidth, collapse = ","), cwp@snthresh,
                     paste(cwp@prefilter, collapse = ","), cwp@mzdiff),
             sprintf("PeakDensityParam: bw=%s minFraction=%s", pdp@bw, pdp@minFraction)),
           session_file)
cat(sprintf("session info -> %s\n", session_file))
