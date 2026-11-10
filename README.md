# MINUT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements.txt)

Max-INtensity Untargeted Transformation: a direct route from LC-HRMS runs to a
sample × feature matrix, without peak picking or retention-time alignment.

Each run is binned on a fixed (m/z, retention time) grid and every bin keeps its most
intense point together with the m/z and RT at which it occurred. The whole transform is
`minut.py` (about 100 lines). `case_studies.py` reruns the three case studies of the
paper from the deposited feature tables, and `extract_minut.py` applies the transform to a
run of your own; the `reproducibility/` notebooks recompute every number of the article.

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

The feature tables (`data/*.pkl`, 350 MB) and the re-extracted COVID-19 matrix behind the MS-level comparison of
the Supporting Information (`data/covid_mz0p5_rt1.npz`, 100 MB) are stored with Git LFS: clone with Git LFS
installed (or run `git lfs pull` after cloning), or download them from the `data/` folder on GitHub.


## Usage

MINUT takes one LC-HRMS run (an mzML or mzXML file) and returns one vector per run, always of
the same length, whatever the instrument did that day. There is nothing to tune beyond the
grid: no peak picking, no alignment, no gap filling. Three steps are enough to go from raw
files to a classifier.

### 1. Transforming your own runs

A grid is six numbers: the m/z range and bin width, the retention-time range and bin width
(`mz_min mz_max mz_width rt_min rt_max rt_width`, RT in minutes):

```bash
python extract_minut.py run.mzML  --mz-range 50 1500 --rt-range 1 30 --mz-width 1 --rt-width 0.5   # your own grid
python extract_minut.py run.mzXML --grid covid --normalization tic  # intensity divided by the TIC instead of the maximum
```

By default only MS1 scans are binned and the intensity block is divided by the per-sample
maximum, as in the article. 

The `.npz` holds the three per-bin arrays (`intensity`, `mz`, `rt`, shape `n_rt × n_mz`), the
flattened `vector` used by the classifiers, the `grid` that produced it and the `normalization`
used, so that a vector can always be traced back to its grid. Keep the same grid for every run
of a cohort: that is what makes the vectors comparable.

### 2. The same thing in Python

```python
import numpy as np
from minut import grid_shape, transform_file

# a grid is six numbers: mz_min, mz_max, mz_width, rt_min, rt_max, rt_width (RT in minutes);
# this one is the command line above: --mz-range 50 1500 --rt-range 1 30 --mz-width 1 --rt-width 0.5
grid = (50.0, 1500.0, 1.0, 1.0, 30.0, 0.5)
vector = transform_file("run.mzML", grid)   # MS1 scans by default; ms_levels=None bins every scan

n_rt, n_mz = grid_shape(grid)               # 58 RT bins × 1450 m/z bins = 84 100 bins
n_bins = n_rt * n_mz
intensity, mz, rt = vector[:n_bins], vector[n_bins:2 * n_bins], vector[2 * n_bins:]
```


The vector is `[intensity | m/z | RT]`: three blocks of `n_bins` values, each block laid out
RT-major (bin `b` is RT row `b // n_mz`, m/z column `b % n_mz`). The intensity block is the
most intense point of each bin divided by the per-sample maximum; the m/z and RT blocks are
the coordinates at which that point was found, min–max scaled with the grid limits so that
every run shares the same frame. Empty bins are 0 in the three blocks. Turning a bin back into
physical coordinates is one line:

```python
b = int(np.argmax(intensity))               # the bin holding the base peak of the run
mz_min, mz_max, mz_w, rt_min, rt_max, rt_w = grid
i_rt, j_mz = divmod(b, n_mz)
print(f"bin {b}: m/z [{mz_min + j_mz * mz_w:g}, {mz_min + (j_mz + 1) * mz_w:g}) × RT [{rt_min + i_rt * rt_w:g}, {rt_min + (i_rt + 1) * rt_w:g}) min")
print(f"retained point: m/z {mz_min + mz[b] * (mz_max - mz_min):.3f}, RT {rt_min + rt[b] * (rt_max - rt_min):.2f} min")
```

```
bin 4796: m/z [496, 497) × RT [2.5, 3) min
retained point: m/z 496.340, RT 2.50 min
```

Two other normalizations of the intensity block are available, `transform_file(path, grid,
normalization="tic")` (total ion current of the binned scans) and `"none"` (raw counts);
the article uses the per-sample maximum.

### 3. From vectors to a classifier

Stack the vectors of a cohort into a matrix (one row per run) and feed it to any classifier;
the article uses a random forest with in-fold selection of the most important descriptors.
Because every column keeps its coordinates, the bins the model relies on can be read back and
checked in the raw data, which is what the `reproducibility/` notebooks do for the three cohorts.

```python
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

files = sorted(Path("my_runs").glob("*.mzML"))
X = np.vstack([transform_file(f, grid) for f in files])
y = np.array([1 if "case" in f.name else 0 for f in files])          # your labels
print(cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0), X, y, cv=5).mean())
```

A few practical points. The grid is the only parameter that matters: start from the RT range
where your chromatography is informative and from a m/z width of the order of your mass
accuracy times a few (the article uses 0.5 to 10 m/z); `python case_studies.py` and the
notebooks show how the bin size was chosen automaticaly on training data only. MINUT does no centroiding:
profile or centroid spectra are binned as they are stored in the file. Intensity descriptors
carry acquisition-batch information, so keep batches balanced across classes or evaluate with
a batch held out (see `docs/downstream_workflow.md`). Once a bin ranks high, treat it as a
region to investigate, not as an identified compound: the same notebooks show how we went
from a bin to extracted-ion chromatograms, isotope ratios and MS/MS evidence.

## Material of the second revision

For the second revision we gathered the code behind every number, table and figure of the article in
`reproducibility/`: one module (`reproducibility/minut.py`, the transform plus the helpers the notebooks share) and one
notebook per case study (`milk.ipynb`, `lung.ipynb`, `covid.ipynb`). Each notebook runs end to end from the
deposited feature tables and writes its tables and figures to `reproducibility/results/`. `docs/downstream_workflow.md` describes what we recommend doing once MINUT has
ranked the bins. The transform itself gained two normalization options: `minut.feature_vector` and
`minut.transform_file` accept `normalization="tic"` or `"none"` besides the per-sample maximum of the article.

## Data

`data/*.pkl` are pandas DataFrames with one row per run. `echantillon` is the run identifier
(the file name for the two public cohorts, the sample number for the milk) and `output_vector`
the MINUT vector `[intensity | m/z | RT]`, each block flattened RT-major and scaled to [0, 1]:
intensity by the per-sample maximum, m/z and RT by the grid limits. The milk table also carries
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
without any filter, and MINUT was applied to every scan of the file: MS$^1$ and MS$^2$ scans
for MTBLS2291 and, for the Waters files of MTBLS6990, the low- and high-energy functions
and the lock-mass reference function. The milk tables come from the centroided MS$^1$ mzXML
files supplied with the samples. `transform_file(path, grid, ms_levels=None)` on such a
file gives back the deposited vector: identically for MTBLS2291; for MTBLS6990 the RT
block and for the milk the intensity block differ by at most 5·10⁻⁸, because the original
run held these two quantities in single precision, everything else being identical
(checked on samples of each cohort). `ms_levels=(1,)`, the default, bins the MS$^1$ scans
only.


## Reference peak-picking tools (Table 5)

The asari, FeatureFinderMetabo and XCMS rows of Table 5 come from `reproducibility/scripts/`:
`extract_asari.py`, `extract_ffm.py` and `extract_xcms.R` extract one feature table per dataset
and tool from the converted raw files, and `bench_eval.py` scores
such a table with the classifier of the Supporting Information (section 2.3). The raw files
(mzML/mzXML) are not distributed here: point `MINUT_DATA_DIR` to the folder that holds them
and list them in a manifest with the columns `sample_name, file_path, label` (the manifests we
used are in `Archive/manifests/`). asari and pyOpenMS are the optional entries of
`requirements.txt`; XCMS needs R ≥ 4.3 with `xcms`, `MsExperiment` and `BiocParallel`. The
tables are written to `outputs/features/` (override with `MINUT_FEATURES_DIR`), together with
the versions of the software that produced them.

```bash
python  reproducibility/scripts/extract_ffm.py   --dataset lung --manifest Archive/manifests/lung.csv
python  reproducibility/scripts/extract_asari.py --dataset lung --manifest Archive/manifests/lung.csv
Rscript reproducibility/scripts/extract_xcms.R   lung Archive/manifests/lung.csv
```

## Archive

`Archive/` holds the material of the first submission as it was deposited: the three
case-study notebooks and their feature tables (`CaseStudy/`), the first reproducibility
notebook (`MINUT_full_reproducibility.ipynb`), the walk-through notebook of the first
revision (`MINUT.ipynb`), the candidate-region statistics of that revision (`regions.py` and
its two tables) and the manifests of the raw files. It is kept
for the record and is not needed to reproduce the article; the numbers of the paper are those
of `case_studies.py` and of the `reproducibility/` notebooks. `Archive/README.md` says what each
file is.

## Citation

If you use MINUT, please cite the article and the github repository (*Cite this
repository* button reads `CITATION.cff`):

> Meo V., Gauriat B., Garnier J.-F., Bertrand D., Tsamba L., Cordella C. Max-INtensity Untargeted
> Transformation (MINUT) for Direct Chemometric Modeling of High-Resolution Mass Spectrometry Data.
> *ACS Omega*, under review.

```bibtex
@software{minut_2026,
  author    = {Meo, Valentin and Gauriat, B{\'e}n{\'e}dicte and Garnier, Jean-Fran{\c{c}}ois and
               Bertrand, Dominique and Tsamba, Lucie and Cordella, Christophe},
  title     = {{MINUT}: Max-INtensity Untargeted Transformation for direct chemometric modeling of HRMS data},
  year      = {2026},
  version   = {1.0.0},
  url       = {https://github.com/valkenzz/MINUT-for-Direct-Chemometric-Modeling-of-HRMS-Spec}
}
```

## License

MIT license (see `LICENSE`). MINUT was developed in the [LARTIC](https://lartic.fsaa.ulaval.ca/equipe)
team, Département des sciences des aliments, Université Laval.
