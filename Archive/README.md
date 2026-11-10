# Archive: material of the first submission and first revision

These files are kept as they were deposited with the first submission (July 2026) and the first revision
(September 2026) and are not needed to reproduce the article. The numbers of the paper are those of `../case_studies.py` and of the three
notebooks in `../reproducibility/`, which supersede everything here. In particular, the compound names used in
these notebooks (PABA, silybin, sphingosine, valine) were withdrawn or revised during the revisions; the
current identities and the raw-data evidence behind them are in `../reproducibility/`.

| File | What it is |
|---|---|
| `CaseStudy/milk/MilkCaseStudy.ipynb`, `milk_10mz.csv` | First analysis of the milk cohort and its feature table (10 m/z × 2 min). |
| `CaseStudy/lung/LungCancer.ipynb`, `dfLung{10,2,0_5}MZ2Min.csv` | First analysis of the lung cohort (three bin sizes). It needs the MetaboLights MTBLS6990 ISA files, now in `../data/metadata/`. |
| `CaseStudy/covid/Covid.ipynb`, `covid_0.5MZ.csv` | First analysis of the COVID-19 cohort. |
| `MINUT_full_reproducibility.ipynb` | First reproducibility notebook: a self-contained copy of the transform, the wrappers of the reference peak-picking tools and, in its section 3, the classifier used to score their feature tables for Table 5 (scikit-learn 1.7.2 at the time, not the pinned environment). |
| `MINUT.ipynb` | Walk-through notebook of the first revision: the transform on a toy grid and on one raw file, then the three classifications and the candidate regions of that version through `../case_studies.py`. Run it from this folder; it imports the modules of the repository root. |
| `regions.py`, `covid_regions.csv`, `milk_paba.csv` | Candidate-region statistics of the first revision (bin 16044 then read as sphingosine d18:1, the valine-mass bins 2436 and 1236, the 138.05 group of bin 183 then read as PABA), with the tables they produce. `python Archive/regions.py` regenerates them from the deposited feature tables. |
| `manifests/{covid,lung,milk}.csv` | Lists of the raw files of each cohort (`sample_name, file_path, label`), the input of `../reproducibility/scripts/extract_*`. |

The feature tables (`*.csv`, pandas pickles despite the extension) are stored with Git LFS. The notebooks
read them from their own folder and resolve the raw files through `MINUT_DATA_DIR`.
