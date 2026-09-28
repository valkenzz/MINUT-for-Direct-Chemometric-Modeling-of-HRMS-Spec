# MINUT

Max-INtensity Untargeted Transformation: a direct route from LC-HRMS runs to a
sample × feature matrix, without peak picking or retention-time alignment.

Each run is binned on a fixed (m/z, retention time) grid and every bin keeps its most
intense point together with the m/z and RT at which it occurred. The whole transform is
`minut.py` (about 100 lines). `case_studies_rerun.py` reruns the three case studies of the
paper from the deposited feature tables; `MINUT.ipynb` walks through both.

MINUT was developed in the [LARTIC](https://lartic.fsaa.ulaval.ca/equipe) team
(Département des sciences des aliments, Université Laval).

## Installation

Python 3.11 with the pinned versions of `requirements.txt`. Random forests differ between
scikit-learn versions, so the numbers below are only reproduced with these versions.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
git lfs install && git lfs pull        # the feature tables in data/ are stored with Git LFS
```

## Reproducing the paper

```
python case_studies_rerun.py           # prints the numbers below and writes them to results/, a few minutes
```

| | Paper | `case_studies_rerun.py` |
|---|---|---|
| Milk, training 10-fold CV accuracy | 0.92 | 0.9219 ± 0.0386 |
| Milk, test set (36 samples): bootstrap accuracy / SD / ROC AUC | 0.95 / 0.04 / 0.97 | 0.9454 / 0.0378 / 0.9733 (34/36 correct) |
| Milk, permutation p-value | 0.0099 | 0.0099 |
| Lung, validation cohort (151 samples): accuracy / ROC AUC | 1.00 / 1.00 | 1.00 / 1.00 |
| Lung, training 10-fold CV accuracy (Table 5, MINUT row) | 1.00 ± 0.00 | 1.00 ± 0.00 |
| COVID-19, stratified 5-fold CV: accuracy / ROC AUC | 0.854 ± 0.100 / 0.866 ± 0.129 | 0.8538 ± 0.0998 / 0.8663 ± 0.1289 |
| Sphingosine-compatible bin 16044: n, Mann–Whitney U, p | 51 (26 controls, 25 severe), 458, 0.013 | same |
| PABA-compatible bin 183: conventional / organic, Fisher p | 105 / 8, 2.4·10⁻¹⁵ | same |

The ablation rows of Table 1 (1D binning, 2D averaging) come from separate pipelines that are not part of this package.

### Protocols

*Milk.* First injection of each sample, batches 20220215 and 20220216 (177 samples), m/z
and RT blocks only, 141/36 split (`random_state=42`). Cross-validation: the 100 most
important features (Gini, forest seed 0) refitted in each fold, classifier seed 2024.
Final model: importance-threshold selection at rank 100 (`threshold_selection`), the
selection of the original analysis behind the test-set numbers. The permutation test is
the one of the original analysis: 100 label permutations of the test set, 5-fold, on the
selected features.

*Lung.* Discovery cohort (85) for training, validation cohort (151) for testing; intensity
and m/z blocks. Cross-validation with the 6 most important features (seed 2024); final
model with the same threshold selection at rank 6, i.e. the seven bins of Table 3
(classifier seed 42).

*COVID-19.* One injection per subject (63: 32 severe, 31 controls). In each of five
stratified folds (`shuffle=True, random_state=2024`) the selector keeps the 1000 features
with the largest Gini importance (1000 trees, seed 0), scores them by the average of a
min–max scaled Mann–Whitney U (controls first) and a Fisher criterion, and keeps the best
100; the classifier is a 100-tree forest (seed 0). Reported values are the mean and SD
over the five held-out folds.

## Data

`data/*.pkl` are pandas DataFrames with one row per run. `echantillon` is the run identifier
(the file name for the two public cohorts, the sample number for the milk) and `output_vector`
the MINUT vector `[intensity | m/z | RT]`, each block flattened RT-major and scaled to [0, 1]:
intensity by the run maximum, m/z and RT by the grid limits. The milk table also carries
`batch` and `categ` (b = organic, c = conventional).

| File | Grid | Source |
|---|---|---|
| `covid_0.5mz_1min.pkl` | m/z 100–700 by 0.5, RT 2–20 min by 1 | MetaboLights [MTBLS2291](https://www.ebi.ac.uk/metabolights/MTBLS2291) |
| `lung_{10,2,0.5}mz_2min.pkl` | m/z 50–1000, RT 2–12 min by 2 | MetaboLights [MTBLS6990](https://www.ebi.ac.uk/metabolights/MTBLS6990) |
| `milk_10mz.pkl` | m/z 100–1000 by 10, RT 2–16 min by 2 | TOFoo project (raw data on request to the corresponding author) |

`covid_injections.csv` and `lung_injections.csv` list the public raw files (URL, SHA-256)
with their subject, group or disease, cohort and batch; the loaders take the labels from them.

### Regenerating the tables

The public tables were computed from mzML/mzXML files written by ProteoWizard `msconvert`
without any filter, and MINUT was applied to every scan of the file: MS1 and MS2 scans
for MTBLS2291 and, for the Waters files of MTBLS6990, the low- and high-energy functions
and the lock-mass reference function. The milk tables come from the centroided MS1 mzXML
files supplied with the samples. `transform_file(path, grid, ms_levels=None)` on such a
file gives back the deposited vector: identically for MTBLS2291; for MTBLS6990 the RT
block and for the milk the intensity block differ by at most 5·10⁻⁸, because the original
run held these two quantities in single precision, everything else being identical
(checked on samples of each cohort). `ms_levels=(1,)`, the default, bins the MS1 scans
only.

## Transforming your own runs

```
python extract_minut.py run.mzXML --grid covid                 # MS1 scans, 0.5 m/z × 1 min
python extract_minut.py run.mzML --grid lung --mz-width 10     # coarser grid
python extract_minut.py run.mzXML --grid covid --all-ms-levels
```

or, in Python, `minut.transform_file(path, grid)`, where `grid` is
`(mz_min, mz_max, mz_width, rt_min, rt_max, rt_width)` with RT in minutes. The transform
does no centroiding: profile or centroid spectra are binned as they are stored in the file.

## Earlier material kept in this repository

The files of the first submission stay in place: `MINUT_full_reproducibility.ipynb`, the
`CaseStudy/` notebooks and their feature tables, `scripts/extract_asari.py`,
`scripts/extract_ffm.py` and `scripts/extract_xcms.R` (feature extraction of the four-tool
comparison), and `manifests/`. The numbers of the paper are the ones produced by
`case_studies_rerun.py` above.

### Raw files and manifests

The first-submission workflow starts from the converted raw files (mzML/mzXML), which are
not distributed here. Point `MINUT_DATA_DIR` to the folder that holds them (the scripts and
notebooks default to a folder named `data`, which in this repository holds the feature tables):

```bash
# Windows PowerShell
$env:MINUT_DATA_DIR = "C:\path\to\raw_files"
# Linux/macOS
export MINUT_DATA_DIR=/path/to/raw_files
```

Each `manifests/<dataset>.csv` lists `sample_name, file_path, label`, where `file_path`
is just the file name, resolved under `MINUT_DATA_DIR`. If your downloaded file names
differ, edit the `file_path` column accordingly.

### First-submission notebook

Launch Jupyter from this folder and run `MINUT_full_reproducibility.ipynb` top to bottom:

```bash
jupyter lab MINUT_full_reproducibility.ipynb
```

- **Sections 0–1** — configuration and a self-contained copy of the MINUT transform.
- **Section 1.a** — MINUT feature extraction (writes `outputs/features/<dataset>_minut.csv`).
- **Section 3** — evaluation of the reference-tool feature tables (asari / FFM / XCMS) with a
  100-tree random forest, 10-fold CV, `random_state=2024`; MINUT is not scored here (its
  numbers come from `case_studies_rerun.py`).
- **Section 4** — extraction-time table.

Set `RUN_MODE = "full"` (section 0) and call `run_all()` (section 5) to process every dataset.

### Peak-picking comparison

Feature extraction with the reference peak pickers, one table per dataset and tool, written
to `outputs/features/` (override with `MINUT_FEATURES_DIR`). asari and pyOpenMS are the
optional entries of `requirements.txt`; XCMS needs R ≥ 4.3 with `xcms`, `MsExperiment` and
`BiocParallel`.

```bash
python  scripts/extract_ffm.py   --dataset lung --manifest manifests/lung.csv
python  scripts/extract_asari.py --dataset lung --manifest manifests/lung.csv
Rscript scripts/extract_xcms.R   lung manifests/lung.csv
```

### First-submission case studies

`CaseStudy/` contains the original downstream analyses, one subfolder per dataset (milk /
PABA, lung / silybin, COVID-19 / sphingosine and valine), each notebook alongside its
feature tables. The milk and COVID-19 notebooks run out of the box; the lung notebook
additionally needs the public [MTBLS6990](https://www.ebi.ac.uk/metabolights/MTBLS6990)
ISA metadata files (`s_MTBLS6990.txt` and
`a_MTBLS6990_LC-MS_untargeted_positive_reverse-phase_metabolite_profiling.txt`), resolved
under `MINUT_DATA_DIR`.

## License and citation

MIT license. Please cite the paper (`CITATION.cff`). Team page: [LARTIC, Université Laval](https://lartic.fsaa.ulaval.ca/equipe).
