# From a ranked bin to a chemical hypothesis: the downstream workflow we recommend

MINUT stops where most peak-picking workflows begin, with a short list of (m/z, RT) bins whose retained
maxima separate the classes. MINUT is first and foremost a classification tool and a tool for prioritizing discriminative regions. A MINUT column is not a peak-table entry: in each sample it holds the most
intense point of the bin, which may be a different ion from one sample to the next. The steps below are
what we do to turn a ranked bin into something a chemist can act on. Steps 1 to 3 need only the deposited
feature tables (the *Downstream inspection* sections of the `reproducibility/` notebooks run them for the most
important bins of each case study); steps 4 to 7 need the raw files (the *Raw data* sections of the
same notebooks).

| Step | Question | Tool / output | Decision |
|---|---|---|---|
| 1. Rank | Which bins carry class information? | Gini importance of the forest fitted on the training partition (`case_studies.py`),  the hybrid raw-U/Fisher score, SHARP, or something else. Keep the rank instability over seeds in mind: 100-tree importances are noisy. | Shortlist; never read a single rank as a biological ordering. |
| 2. Inspect coordinates | Does the column hold the same ion in every sample? | Per-sample (m/z, RT, I) of the bin: occupancy, m/z spread (ppm), RT spread, dominant coordinate group with its support *D* = \|G\|/n and consistency *C* = \|G\|/\|P\| (full spans 5 ppm × 15 s or 10 ppm × 30 s). | *C* ≥ 0.9: one dominant ion, go to step 4. Otherwise sub-cluster (step 3). |
| 3. Sub-cluster | Which sub-population of points is discriminative? | DBSCAN on m/z (eps 0.01, min 3) and, if RT is retained, on RT; class counts per sub-cluster (a Fisher exact test is descriptive after selection). | Treat each sub-cluster as a separate candidate; samples whose retained maximum is another ion are not evidence of absence. |
| 4. Targeted XIC | Is the candidate ion present in all samples, including those where it lost the bin? | Extracted-ion chromatograms at the cluster mass (5 and 10 ppm) in every raw file; area, apex RT; class ratios pooled and within each acquisition batch (SI sections 5.4 and 7.4, Tables S9 and S25; `milk.ipynb`, section 8; `covid.ipynb`, sections 6 and 7). | Intensity differences that vanish within mixed batches are acquisition effects, not biology. |
| 5. Isotopes / adducts / in-source | Is the formula hypothesis compatible? | M+1/M (and M+2/M where the signal allows) in the same ppm windows against the expectation of the formula, on the scans above 50 % of the apex (SI section 2.4; Table S25 for the COVID-19 candidates); search for co-eluting adduct and in-source fragment masses (Tables S8, S12 and S17). | Incompatible isotope ratios: drop the name, keep the mass — after checking that the M+1 peak lies above the centroiding floor of the converted files (SI sections 5.4 and 7.4). |
| 6. MS/MS and annotation | What is it? | Precursor-centred MS$^2$ within the RT window, library matching (MassBank, GNPS, SIRIUS), annotation tools (MZmine 3, MS-DIAL 4, Compound Discoverer, Progenesis QI); state the evidence behind each name (co-eluting adducts and fragments, isotope ratios, MS/MS match, authentic standard). | Mass agreement alone is a mass-compatible hypothesis, not an identification. |
| 7. Validate | Is the association reproducible? | Batch-aware evaluation (leave-one-batch-out; `covid.ipynb`, section 3), permutation null of the whole protocol (`lung.ipynb`, section 4), independent cohort, authentic standard and targeted quantification. | Only then call the candidate a biomarker. |

## Practical recommendations

* Report *D*, *C* and the sub-cluster class counts for every bin you discuss; they tell the reader how much
  of the cohort the chemical reading applies to.
* Keep the raw files. Everything after step 3 is a targeted re-extraction, which MINUT does not replace.
* Expect intensity descriptors to carry acquisition-batch information. When classes and batches are
  confounded, prefer the coordinate descriptors (m/z, RT) for classification and evaluate with the batch
  held out.
