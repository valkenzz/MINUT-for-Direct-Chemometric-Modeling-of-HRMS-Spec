## Results of the article that are not recomputed here

* The ablation rows of Table 1 (2D averaging, 1D binning) and panel A of Figure S1 came, at first submission, from
  pipelines that were never part of the deposited package; only the MINUT row is recomputed
  (`milk.ipynb`, section 2).
* The asari, FeatureFinderMetabo and XCMS rows of Table 5 come from `scripts/`, `../Archive/manifests/` and
  section 3 of `../Archive/MINUT_full_reproducibility.ipynb`, in the environment recorded there
  (scikit-learn 1.7.2), and their feature tables were not archived. They can be regenerated from the public raw files
  with those scripts, at the cost of hours per tool and cohort.
* The analytical checks of the Supporting Information (section 4, Tables S1–S5) come from a separate pyOpenMS
  workflow run during the first revision, outside this package. The coordinate check of the milk table (section 5.2 of the Supporting
  Information) is recomputed independently in `milk.ipynb` (section 7), and the full-MS rows of Table S21
  read one of those matrices as described above.
