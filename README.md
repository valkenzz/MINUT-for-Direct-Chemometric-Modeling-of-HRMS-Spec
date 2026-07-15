# MINUT — Reproducibility package

This folder contains everything needed to reproduce the results of the MINUT paper.
The MINUT method itself is implemented **inline** in the notebook (plain NumPy + optional
Numba) — there is **no dependency on any proprietary software** and **no machine-specific path**.

## Contents

```
source_code/
├── MINUT_full_reproducibility.ipynb   # main notebook — run this
├── requirements.txt                   # Python dependencies
├── manifests/
│   ├── covid.csv                       # sample_name, file_path, label
│   ├── lung.csv
│   └── milk.csv
├── scripts/                            # reference peak-picker baselines (optional)
│   ├── extract_ffm.py                  # pyOpenMS FeatureFinderMetabo
│   ├── extract_asari.py                # asari
│   └── extract_xcms.R                  # XCMS (R)
└── CaseStudy/                          # downstream case studies (biomarker analysis)
    ├── covid/
    │   ├── Covid.ipynb
    │   ├── covid_0.5MZ.csv              # bundled MINUT feature files
    │   ├── covid_5MZ.csv
    │   └── covid_10MZ.csv
    ├── lung/
    │   ├── LungCancer.ipynb
    │   ├── dfLung0_5MZ2Min.csv          # bundled MINUT feature files
    │   ├── dfLung2MZ2Min.csv
    │   └── dfLung10MZ2Min.csv
    └── milk/
        ├── MilkCaseStudy.ipynb
        └── milk_10mz.csv                # bundled MINUT feature file
```

## 1. Environment

```bash
python -m venv .venv && . .venv/Scripts/activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Recommended: Python 3.10+. Numba is optional (accelerates the binning; identical results).
The XCMS baseline additionally needs R ≥ 4.3 with `xcms`, `MsExperiment`, `BiocParallel`.

## 2. Get the raw data (public)

| Dataset | Repository | Notes |
|---------|-----------|-------|
| COVID-19 plasma | MetaboLights **MTBLS2291** | UPLC-HRMS (`.mzXML`) |
| Lung cancer / TB serum | MetaboLights **MTBLS6990** | UPLC-Q-TOF (`.mzML`) |
| Milk (organic vs conventional) | private (TOFOO project) | not publicly available |

Download the raw files and put them in a single folder, then point the pipeline to it:

```bash
# Windows PowerShell
$env:MINUT_DATA_DIR = "C:\path\to\raw_files"
# Linux/macOS
export MINUT_DATA_DIR=/path/to/raw_files
```

Each `manifests/<dataset>.csv` lists `sample_name, file_path, label`, where `file_path`
is just the **file name**; files are resolved under `MINUT_DATA_DIR`. If your downloaded
file names differ, edit the `file_path` column accordingly.

## 3. Run

Launch Jupyter **from this folder** and run the notebook top to bottom:

```bash
jupyter lab MINUT_full_reproducibility.ipynb
```

- **Section 0–1** — configuration and the self-contained MINUT transform.
- **Section 1.a** — MINUT feature extraction (fast; writes `outputs/features/<dataset>_minut.csv`).
- **Section 3** — leakage-free evaluation of the reference tools (asari/FFM/XCMS), `random_state=2024`; MINUT is not scored here (its accuracy is in the `CaseStudy/` notebooks).
- **Section 4** — extraction-time table.

Set `RUN_MODE = "full"` (section 0) and call `run_all()` (section 5) to reproduce every dataset.

## 4. Reference baselines (optional)

The three peak-pickers in `scripts/` are run at their **published defaults** (no per-dataset
tuning). They require the extra dependencies above and can take tens of minutes to hours on
full cohorts. They are invoked automatically by `run_all(...)`, or manually, e.g.:

```bash
python scripts/extract_ffm.py   --dataset lung --manifest manifests/lung.csv
python scripts/extract_asari.py --dataset lung --manifest manifests/lung.csv
Rscript  scripts/extract_xcms.R  lung manifests/lung.csv
```

Outputs go to `outputs/features/` (override with `MINUT_FEATURES_DIR`).

## Case studies (biomarker analysis)

`CaseStudy/` contains the downstream analyses, one subfolder per dataset (milk / PABA,
lung / Silybin, COVID-19 / C19-sphingosine & valine) starting from the MINUT features. Each
subfolder holds its notebook (`CaseStudy/milk/MilkCaseStudy.ipynb`,
`CaseStudy/lung/LungCancer.ipynb`, `CaseStudy/covid/Covid.ipynb`) alongside its bundled MINUT
feature files. The milk study runs out of the box; the lung study additionally needs the
public MTBLS6990 ISA metadata files.

## 5. Reproducibility notes

- MINUT binning is max-intensity; Numba only accelerates it (results identical to pure Python,
  m/z / RT differ only by float32 rounding, well below 1 ppm).
- Only `MINUT_DATA_DIR` (and optionally `MINUT_FEATURES_DIR`) are environment-specific.
