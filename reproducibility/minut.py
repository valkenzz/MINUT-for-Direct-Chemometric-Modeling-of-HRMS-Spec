"""MINUT (Max-INtensity Untargeted Transformation) and the helpers shared by the three notebooks.

The first section is the transform itself. It is the same code as minut.py at the root of the
repository, repeated here so that this folder runs on its own. The rest is what the case studies need
and would otherwise be copied from one notebook to the next: the deposited cohorts, the feature
selectors and the random-forest protocol of the article, a few tools to read a bin back into physical
coordinates, and readers for the raw files.

The deposited tables are read from ../data. Raw files are optional: when the environment variable
MINUT_RAW points to a folder with covid/, milk/, lung*/ and maf/ sub-folders, the raw-data sections of
the notebooks run; otherwise they fall back on the per-sample tables left by the last pass.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data"
RAW = Path(os.environ.get("MINUT_RAW", REPO / "raw"))
RESULTS = HERE / "results"

# ---------------------------------------------------------------------------
# The transform (same code as ../minut.py)
# ---------------------------------------------------------------------------

# (mz_min, mz_max, mz_width, rt_min, rt_max, rt_width); m/z in Th, RT in minutes.
GRIDS = {
    "milk": (100.0, 1000.0, 10.0, 2.0, 16.0, 2.0),
    "lung": (50.0, 1000.0, 2.0, 2.0, 12.0, 2.0),
    "covid": (100.0, 700.0, 0.5, 2.0, 20.0, 1.0),
}


def grid_shape(grid):
    """Number of (RT, m/z) bins of a grid. A last, incomplete bin is dropped."""
    mz_min, mz_max, mz_width, rt_min, rt_max, rt_width = grid
    return int((rt_max - rt_min) / rt_width), int((mz_max - mz_min) / mz_width)


@njit
def _bin_max(rt, mz, intensity, mz_min, mz_max, mz_width, rt_min, rt_max, rt_width, max_i, max_mz, max_rt):
    n_rt, n_mz = max_i.shape
    inv_rt, inv_mz = 1.0 / rt_width, 1.0 / mz_width
    for k in range(intensity.size):
        if rt[k] < rt_min or rt[k] >= rt_max or mz[k] < mz_min or mz[k] >= mz_max:
            continue
        i = int((rt[k] - rt_min) * inv_rt)
        j = int((mz[k] - mz_min) * inv_mz)
        if i >= n_rt or j >= n_mz:
            continue
        # strictly greater on purpose: of two equal maxima, the first point seen is the one we keep
        if intensity[k] > max_i[i, j]:
            max_i[i, j] = intensity[k]
            max_mz[i, j] = mz[k]
            max_rt[i, j] = rt[k]


def minut(spectra, grid):
    """Bin one run and keep, in each bin, its most intense point.

    spectra yields one (rt_minutes, mz_array, intensity_array) per scan. We return three (n_rt, n_mz)
    arrays, the intensity, m/z and RT of the retained point. A bin that received no point has
    intensity 0 and nan coordinates.
    """
    mz_min, mz_max, mz_width, rt_min, rt_max, rt_width = grid
    shape = grid_shape(grid)
    max_i = np.zeros(shape)
    max_mz = np.full(shape, np.nan)
    max_rt = np.full(shape, np.nan)
    for rt, mz, intensity in spectra:
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        rt_array = np.full(mz.size, float(rt))
        _bin_max(rt_array, mz, intensity, mz_min, mz_max, mz_width, rt_min, rt_max, rt_width, max_i, max_mz, max_rt)
    return max_i, max_mz, max_rt


def feature_vector(max_i, max_mz, max_rt, grid, normalization="max", tic=None):
    """Flatten the three arrays into one vector [intensity | m/z | RT], each block RT-major, scaled to [0, 1].

    The intensity block is divided by the maximum of the run ("max", what the article uses), by the
    total ion current of the binned scans ("tic") or left as acquired ("none"). m/z and RT are min-max
    scaled with the limits of the grid, so that all runs share the same coordinate frame. An empty bin
    is 0 in every block.
    """
    mz_min, mz_max, _, rt_min, rt_max, _ = grid
    if normalization == "max":
        intensity = max_i / max_i.max() if max_i.max() > 0 else max_i
    elif normalization == "tic":
        if tic is None or tic <= 0:
            raise ValueError("normalization='tic' requires the total ion current of the run")
        intensity = max_i / tic
    elif normalization == "none":
        intensity = max_i
    else:
        raise ValueError(f"unknown normalization {normalization!r}; use 'max', 'tic' or 'none'")
    mz = np.where(np.isnan(max_mz), mz_min, max_mz)
    rt = np.where(np.isnan(max_rt), rt_min, max_rt)
    return np.concatenate([intensity.ravel(), ((mz - mz_min) / (mz_max - mz_min)).ravel(),
                           ((rt - rt_min) / (rt_max - rt_min)).ravel()])


def read_spectra(path, ms_levels=(1,)):
    """Yield (rt_minutes, mz, intensity) for the scans of an mzML or mzXML file whose MS level is in ms_levels (None keeps every scan)."""
    from pyteomics import mzml, mzxml

    path = Path(path)
    if path.suffix.lower() == ".mzml":
        with mzml.read(str(path)) as reader:
            for scan in reader:
                if ms_levels is not None and scan["ms level"] not in ms_levels:
                    continue
                rt = scan["scanList"]["scan"][0]["scan start time"]
                minutes = float(rt) / 60 if getattr(rt, "unit_info", "minute") == "second" else float(rt)
                yield minutes, scan["m/z array"], scan["intensity array"]
    elif path.suffix.lower() == ".mzxml":
        with mzxml.read(str(path)) as reader:   # pyteomics converts retentionTime to minutes
            for scan in reader:
                if ms_levels is not None and scan["msLevel"] not in ms_levels:
                    continue
                yield float(scan["retentionTime"]), scan["m/z array"], scan["intensity array"]
    else:
        raise ValueError(f"expected an .mzML or .mzXML file, got {path.name}")


def transform_file(path, grid, ms_levels=(1,), normalization="max"):
    """MINUT vector of one run, straight from the file (see feature_vector for the layout)."""
    tic = 0.0

    def spectra():
        nonlocal tic
        for rt, mz, intensity in read_spectra(path, ms_levels):
            tic += float(np.asarray(intensity, dtype=float).sum())
            yield rt, mz, intensity

    max_i, max_mz, max_rt = minut(spectra(), grid)
    return feature_vector(max_i, max_mz, max_rt, grid, normalization=normalization, tic=tic)


# ---------------------------------------------------------------------------
# Deposited cohorts
# ---------------------------------------------------------------------------

def milk_table():
    """The deposited milk table as such: 199 records, batch read as text, categ 'b' = organic and 'c' = conventional."""
    df = pd.read_pickle(DATA / "milk_10mz.pkl")
    df["batch"] = df["batch"].astype(str)
    return df


def milk_matrix(df):
    """m/z and RT blocks of the milk vectors (the intensity block is left out, as in the article) and the labels, organic = 1."""
    n_bins = int(np.prod(grid_shape(GRIDS["milk"])))
    X = np.stack(df["output_vector"].to_list())[:, n_bins:]
    y = (df["categ"] == "b").to_numpy().astype(int)
    return X, y


def load_milk():
    """The milk cohort of the article: first injection of each sample, batches 20220215 and 20220216 only
    (177 samples), with its frozen 141/36 split."""
    df = milk_table().drop_duplicates("echantillon")
    df = df[df["batch"].isin(["20220215", "20220216"])]
    X, y = milk_matrix(df)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    return X, y, train, test


def lung_table(mz_width=2.0):
    """All deposited lung runs (389: clinical, healthy controls, QC, blanks) joined to the MetaboLights ISA metadata."""
    s = pd.read_csv(DATA / "metadata" / "s_MTBLS6990.txt", sep="\t")
    a = pd.read_csv(DATA / "metadata" / "a_MTBLS6990_LC-MS_untargeted_positive_reverse-phase_metabolite_profiling.txt", sep="\t")
    meta = s.merge(a[["Sample Name", "Raw Spectral Data File"]], on="Sample Name")
    meta["run"] = meta["Raw Spectral Data File"].astype(str).str.split("/").str[-1].str.split(".").str[0]
    df = pd.read_pickle(DATA / f"lung_{mz_width:g}mz_2min.pkl")
    df["run"] = df["echantillon"].str.split(".").str[0]
    df = df.merge(meta[["run", "Factor Value[Disease]", "Factor Value[Cohort]", "Characteristics[Sample type]"]], on="run", how="left")
    df = df.rename(columns={"Factor Value[Disease]": "disease", "Factor Value[Cohort]": "cohort", "Characteristics[Sample type]": "sample_type"})
    df["date"] = df["run"].str.extract(r"lcy(\d{8})_")[0]
    df["run_index"] = df["run"].str.extract(r"_(\d+)$")[0].astype(int)
    return df


def lung_blocks(df, mz_width=2.0, n_blocks=2):
    """First n_blocks of the lung vectors: [intensity | m/z] with n_blocks = 2, the representation of the article; 3 gives the full triplet."""
    grid = GRIDS["lung"][:2] + (mz_width,) + GRIDS["lung"][3:]
    n_bins = int(np.prod(grid_shape(grid)))
    return np.stack(df["output_vector"].to_list())[:, :n_blocks * n_bins]


def load_lung(mz_width=2.0):
    """The 236 clinical sera (lung cancer = 1, tuberculosis = 0), intensity and m/z blocks, with the predefined discovery (85) and validation (151) cohorts."""
    runs = pd.read_csv(DATA / "lung_injections.csv", dtype=str)
    runs = runs[runs["sample_type"] == "clinical"]
    df = pd.read_pickle(DATA / f"lung_{mz_width:g}mz_2min.pkl")
    df["sample_name"] = df["echantillon"].str.split(".").str[0]
    df = runs.merge(df, on="sample_name")
    X = lung_blocks(df, mz_width)
    y = (df["disease"] == "lung cancer").to_numpy().astype(int)
    train = np.flatnonzero(df["cohort"] == "discovery")
    test = np.flatnonzero(df["cohort"] == "validation")
    return X, y, train, test, df.reset_index(drop=True)


def load_covid():
    """The 63 COVID-19 plasma samples (severe = 1, control = 0), one injection per subject, with the acquisition batch of each run (date prefix of the file name)."""
    runs = pd.read_csv(DATA / "covid_injections.csv", dtype=str)
    df = pd.read_pickle(DATA / "covid_0.5mz_1min.pkl")
    df["sample_name"] = df["echantillon"].str.split(".").str[0]
    df = df.merge(runs[["sample_name", "subject_id", "disease", "batch"]], on="sample_name")
    X = np.stack(df["output_vector"].to_list())
    y = (df["disease"] == "Severe COVID-19").to_numpy().astype(int)
    return X, y, df[["sample_name", "subject_id", "disease", "batch"]].reset_index(drop=True)


def decoded_blocks(X, grid):
    """Split a feature matrix into its intensity block and its m/z and RT blocks converted back to Th and minutes."""
    mz_min, mz_max, _, rt_min, rt_max, _ = grid
    nb = int(np.prod(grid_shape(grid)))
    return X[:, :nb], mz_min + (mz_max - mz_min) * X[:, nb:2 * nb], rt_min + (rt_max - rt_min) * X[:, 2 * nb:3 * nb]


# =============================================================================== selection and classification

class GiniSelector(BaseEstimator, TransformerMixin):
    """Keep the k descriptors with the largest Gini importance of a 100-tree forest. Used inside a Pipeline, so the selection is refitted in every training fold."""

    def __init__(self, k, random_state=0):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=100, random_state=self.random_state).fit(X, y)
        self.indices_ = np.argsort(forest.feature_importances_)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.indices_]


class HybridSelector(BaseEstimator, TransformerMixin):
    """The COVID-19 selector. A 1000-tree forest pre-ranks the descriptors; the best pre_rank of them are
    then scored by the mean of a min-max scaled Mann-Whitney U (controls as the first sample) and a
    Fisher criterion, and the k highest are kept. Ties fall back on the Gini order."""

    def __init__(self, k=100, pre_rank=1000, n_jobs=4):
        self.k = k
        self.pre_rank = pre_rank
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from scipy.stats import mannwhitneyu
        forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=self.n_jobs).fit(X, y)
        top = np.argsort(forest.feature_importances_)[::-1][:self.pre_rank]
        Xt = X[:, top]
        u = np.array([mannwhitneyu(Xt[y == 0, j], Xt[y == 1, j], alternative="two-sided").statistic for j in range(Xt.shape[1])])
        fisher = (Xt[y == 1].mean(0) - Xt[y == 0].mean(0)) ** 2 / (Xt[y == 1].var(0) + Xt[y == 0].var(0) + 1e-8)
        scale = lambda v: MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel()
        score = (scale(u) + scale(fisher)) / 2
        order = sorted(range(len(top)), key=lambda j: score[j], reverse=True)
        self.indices_ = top[order[:self.k]]
        return self

    def transform(self, X):
        return X[:, self.indices_]


def threshold_selection(X, y, k, random_state):
    """Selection used for the final models in the original notebooks: a descriptor is kept when its
    importance reaches the (k+1)-th largest value. The threshold is inclusive, which is why k+1
    descriptors come out (101 for the milk, 7 for the lung); we kept it that way to reproduce the numbers."""
    importance = RandomForestClassifier(n_estimators=100, random_state=random_state).fit(X, y).feature_importances_
    return importance >= np.sort(importance)[::-1][k]


def holdout(X, y, train, test, keep, seed, n_boot=1000):
    """Fit a 100-tree forest on the training samples with the selected descriptors and score the held-out samples, with a percentile bootstrap of the test accuracy."""
    model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X[train][:, keep], y[train])
    pred = model.predict(X[test][:, keep])
    proba = model.predict_proba(X[test][:, keep])[:, 1]
    boot = [accuracy_score(*resample(y[test], pred, random_state=i)) for i in range(n_boot)]
    return model, pred, proba, dict(test_accuracy=accuracy_score(y[test], pred), test_auc=roc_auc_score(y[test], proba),
                                    bootstrap_accuracy=float(np.mean(boot)), bootstrap_sd=float(np.std(boot)), n_features=int(keep.sum()))


def wilson(k, n, z=1.96):
    """Wilson 95 % interval of a proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def classification_summary(y_true, y_pred, y_proba=None, positive="organic", negative="conventional"):
    """Confusion matrix, accuracy, balanced accuracy, class-wise sensitivities and their Wilson intervals (class 1 is the positive one)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    out = dict(n=len(y_true), n_negative=tn + fp, n_positive=tp + fn, tn=tn, fp=fp, fn=fn, tp=tp,
               accuracy=accuracy_score(y_true, y_pred), balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
               majority_baseline=max((y_true == 0).mean(), (y_true == 1).mean()))
    out[f"sensitivity_{positive}"] = tp / (tp + fn) if tp + fn else np.nan
    out[f"sensitivity_{negative}"] = tn / (tn + fp) if tn + fp else np.nan
    out["accuracy_wilson95"] = wilson(tp + tn, len(y_true))
    out[f"sensitivity_{positive}_wilson95"] = wilson(tp, tp + fn)
    out[f"sensitivity_{negative}_wilson95"] = wilson(tn, tn + fp)
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out


def evaluate_splits(pipeline, X, y, splits, groups=None):
    """Fit and predict a pipeline on each (train, test) split. Returns the per-split table, the out-of-fold probabilities and the out-of-fold predictions."""
    rows, oof_proba, oof_pred = [], np.full(len(y), np.nan), np.full(len(y), -1)
    for k, (train, test) in enumerate(splits):
        fitted = pipeline.fit(X[train], y[train])
        proba = fitted.predict_proba(X[test])[:, 1]
        pred = fitted.predict(X[test])
        oof_proba[test], oof_pred[test] = proba, pred
        row = dict(fold=k, n_train=len(train), n_test=len(test), n_test_control=int((y[test] == 0).sum()),
                   n_test_severe=int((y[test] == 1).sum()), accuracy=accuracy_score(y[test], pred),
                   auc=roc_auc_score(y[test], proba) if len(np.unique(y[test])) == 2 else np.nan)
        if groups is not None:
            row["held_out_group"] = ";".join(sorted(set(map(str, groups[test]))))
        rows.append(row)
    return pd.DataFrame(rows), oof_proba, oof_pred


def summarize_folds(table, label):
    """Mean and SD (population SD, ddof = 0) of the fold accuracies and AUCs, the per-fold values being kept as text."""
    acc, auc = table["accuracy"], table["auc"].dropna()
    return dict(protocol=label, n_folds=len(table), accuracy=acc.mean(), accuracy_sd=acc.std(ddof=0),
                auc=auc.mean() if len(auc) else np.nan, auc_sd=auc.std(ddof=0) if len(auc) else np.nan,
                n_folds_with_auc=len(auc), fold_accuracy=";".join(f"{v:.4f}" for v in acc),
                fold_auc=";".join(f"{v:.4f}" for v in table["auc"]))


# ---------------------------------------------------------------------------
# Reading a bin
# ---------------------------------------------------------------------------

def bin_coordinates(X, bin_id, grid, components=("intensity", "mz", "rt")):
    """Intensity, m/z (Th) and RT (min) of one bin for every sample of a feature matrix."""
    mz_min, mz_max, _, rt_min, rt_max, _ = grid
    n_bins = int(np.prod(grid_shape(grid)))
    block = {name: X[:, i * n_bins + bin_id] for i, name in enumerate(components)}
    out = pd.DataFrame(index=range(len(X)))
    if "intensity" in block:
        out["intensity"] = block["intensity"]
    if "mz" in block:
        out["mz"] = block["mz"] * (mz_max - mz_min) + mz_min
    if "rt" in block:
        out["rt"] = block["rt"] * (rt_max - rt_min) + rt_min
    return out


def bin_limits(bin_id, grid):
    """Limits (mz_lo, mz_hi, rt_lo, rt_hi) of a bin, bins being numbered RT-major."""
    mz_min, _, mz_width, rt_min, _, rt_width = grid
    rt_bin, mz_bin = divmod(bin_id, grid_shape(grid)[1])
    return (mz_min + mz_bin * mz_width, mz_min + (mz_bin + 1) * mz_width, rt_min + rt_bin * rt_width, rt_min + (rt_bin + 1) * rt_width)


def bin_label(bin_id, grid):
    """Readable label of a bin, e.g. 'm/z [130,140) RT [6,8) min'."""
    return "m/z [%g,%g) RT [%g,%g) min" % bin_limits(bin_id, grid)


def bin_index(grid, mz, rt):
    """Index of the bin that holds (mz, rt), or None when the point falls outside the grid."""
    mz_min, mz_max, mz_w, rt_min, rt_max, rt_w = grid
    n_mz = int(round((mz_max - mz_min) / mz_w))
    if not (rt_min <= rt < rt_max and mz_min <= mz < mz_max):
        return None
    return int((rt - rt_min) / rt_w) * n_mz + int((mz - mz_min) / mz_w)


def mass_cluster(mz, target, present, eps=0.01, min_samples=3):
    """Members of the DBSCAN m/z cluster that contains the point nearest to the target mass; present marks the samples where the bin is non-empty."""
    labels = np.full(len(mz), -2)
    labels[present] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mz[present, None])
    nearest = np.flatnonzero(present)[np.argmin(np.abs(mz[present] - target))]
    return labels == labels[nearest]


def rank_of(importances, feature):
    """Rank of one descriptor among the importances, 1 for the largest."""
    return int((importances > importances[feature]).sum()) + 1


def dominant_group(mz, rt_min, ppm, rt_s):
    """Size of the largest group of points that fit in one rectangle of full span ppm (relative) by rt_s seconds. Exhaustive search over the corners, so quadratic, which is fine for a few hundred points."""
    rt = rt_min * 60.0
    best = 0
    for i in range(len(mz)):
        for j in range(len(mz)):
            inside = (mz >= mz[i]) & (mz <= mz[i] * (1 + ppm)) & (rt >= rt[j]) & (rt <= rt[j] + rt_s)
            best = max(best, int(inside.sum()))
    return best


def inspect_bin(coords, y, bin_id, grid, class_names, ppm, rt_s):
    """What one would want to know about a bin before naming it: occupancy, dispersion of the retained
    maxima, dominant coordinate group (support D = group / n, consistency C = group / present), DBSCAN
    sub-clusters along m/z and a one-line flag. coords may lack the 'rt' column (lung representation)."""
    n = len(y)
    present = coords["intensity"].to_numpy() > 0 if "intensity" in coords else np.ones(n, bool)
    P = int(present.sum())
    out = dict(bin=bin_id, bin_limits=bin_label(bin_id, grid), occupancy=P / n, n_present=P)
    if P == 0:
        return out
    mzp = coords["mz"].to_numpy()[present]
    out.update(mz_mean=mzp.mean(), mz_sd_ppm=mzp.std() / mzp.mean() * 1e6, mz_range_ppm=(mzp.max() - mzp.min()) / mzp.mean() * 1e6)
    if "rt" in coords:
        rtp = coords["rt"].to_numpy()[present]
        out.update(rt_mean_min=rtp.mean(), rt_sd_min=rtp.std(), rt_range_min=rtp.max() - rtp.min())
        G = dominant_group(mzp, rtp, ppm, rt_s)
    else:
        G = max(int(((mzp >= m) & (mzp <= m * (1 + ppm))).sum()) for m in mzp)
    out.update(support_D=G / n, consistency_C=G / P)
    labels = DBSCAN(eps=0.01, min_samples=3).fit_predict(mzp[:, None])
    clusters = []
    for lab in sorted(set(labels) - {-1}):
        sel = labels == lab
        ysel = y[present][sel]
        counts = "/".join(f"{int((ysel == k).sum())} {name}" for k, name in enumerate(class_names))
        clusters.append(f"m/z {mzp[sel].mean():.5f} (n={sel.sum()}: {counts})")
    out["n_mz_subclusters"] = len(clusters)
    out["n_noise_points"] = int((labels == -1).sum())
    out["mz_subclusters"] = " | ".join(clusters)
    ydom = y[present]
    out["class_counts_present"] = "/".join(f"{int((ydom == k).sum())} {name}" for k, name in enumerate(class_names))
    if out["occupancy"] < 0.5:
        out["flag"] = "sparse: check whether absence is informative, then XIC"
    elif out["consistency_C"] >= 0.9:
        out["flag"] = "single dominant ion: proceed to XIC / isotopes / MS2" if "rt" in coords else \
            "single dominant ion (m/z only; RT not retained): proceed to XIC / isotopes / MS2"
    elif len(clusters) >= 2:
        out["flag"] = "mixed column: sub-cluster before any chemical reading"
    else:
        out["flag"] = "dispersed coordinates: inspect per-sample points"
    return out


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------

# exact masses (Da) of the adduct partners; the electron mass is included in the proton
H, Na, K, NH4 = 1.00727646688, 22.98922, 38.963158, 18.03383
H2O, D13C = 18.010565, 1.0033548
PPM = 5e-6
MIXED_BATCHES = ["13092020", "15092020", "20092020", "26092020"]   # COVID acquisition batches holding both classes
PPG_UNIT, PPG_END = 58.04186, 18.010565                          # polypropylene glycol H(OC3H6)nOH


def read_scans(path, rt_lo=-np.inf, rt_hi=np.inf, ms_level=1, function=None):
    """Yield (rt_min, mz, intensity) for the scans of one MS level inside an RT bin. For the Waters
    mzML files, function selects the acquisition function (1 low energy, 2 high energy, 3 lock mass).
    Scans come in acquisition order, so we stop reading once rt_hi is passed."""
    from pyteomics import mzml, mzxml

    path = Path(path)
    if path.suffix.lower() == ".mzml":
        with mzml.read(str(path), use_index=False) as reader:
            for s in reader:
                if s.get("ms level") != ms_level:
                    continue
                if function is not None and not s["id"].startswith(f"function={function} "):
                    continue
                rt = s["scanList"]["scan"][0]["scan start time"]
                rt = float(rt) / 60 if getattr(rt, "unit_info", "minute") == "second" else float(rt)
                if rt < rt_lo:
                    continue
                if rt > rt_hi:
                    break
                yield rt, np.asarray(s["m/z array"], float), np.asarray(s["intensity array"], float)
    else:
        with mzxml.read(str(path), use_index=False) as reader:
            for s in reader:
                if s.get("msLevel") != ms_level:
                    continue
                rt = float(s["retentionTime"])
                if rt < rt_lo:
                    continue
                if rt > rt_hi:
                    break
                yield rt, np.asarray(s["m/z array"], float), np.asarray(s["intensity array"], float)


def peak(mz, it, target, ppm=PPM):
    """Most intense centroid within +/- ppm of target, 0 when there is none."""
    sel = np.abs(mz - target) <= target * ppm
    return float(it[sel].max()) if sel.any() else 0.0


def peak_mz(mz, it, target, ppm=PPM):
    """Same as peak, but also returns the m/z of that centroid (nan when there is none)."""
    sel = np.abs(mz - target) <= target * ppm
    if not sel.any():
        return 0.0, np.nan
    i = np.flatnonzero(sel)[np.argmax(it[sel])]
    return float(it[i]), float(mz[i])


def bin_maximum(mz, it, mz_lo, mz_hi):
    """Intensity and m/z of the most intense centroid of a scan inside [mz_lo, mz_hi)."""
    sel = (mz >= mz_lo) & (mz < mz_hi)
    if not sel.any():
        return 0.0, np.nan
    j = np.flatnonzero(sel)[np.argmax(it[sel])]
    return float(it[j]), float(mz[j])


def formula_mass(formula):
    """Monoisotopic mass of a formula, written with or without charge signs and separators."""
    from pyteomics import mass
    return mass.calculate_mass(formula=re.sub(r"[^A-Za-z0-9]", "", formula))


def iso_expect(formula, shift=1):
    """Expected (M+shift)/M abundance ratio of a formula, from its natural isotopologue distribution."""
    from pyteomics import mass
    dist = {}
    m0 = mass.calculate_mass(formula=formula)
    for comp, p in mass.isotopologues(formula=formula, report_abundance=True, overall_threshold=1e-4):
        k = round(mass.calculate_mass(comp) - m0)
        dist[k] = dist.get(k, 0) + p
    return dist.get(shift, 0) / dist[0]


def ppg_adducts(n_min=3, n_max=20):
    """(n, adduct, m/z) of the polypropylene-glycol oligomer ions H(OC3H6)nOH, n from n_min to n_max."""
    return [(n, a, n * PPG_UNIT + PPG_END + d) for n in range(n_min, n_max + 1) for a, d in (("H", H), ("NH4", NH4), ("Na", Na), ("K", K))]


def match_ppg(mz, ppm=10e-6, table=None):
    """Label of the PPG ion within ppm of mz, or an empty string."""
    table = table or ppg_adducts()
    best = min(table, key=lambda r: abs(r[2] - mz))
    return f"PPG{best[0]} [M+{best[1]}]+" if abs(best[2] - mz) <= mz * ppm else ""


def van_elteren(values, is_case, strata, levels=None):
    """Stratified Wilcoxon rank-sum test with van Elteren weights. Returns z and the two-sided normal p-value, corrected for ties."""
    values = np.asarray(values, float); is_case = np.asarray(is_case, bool); strata = np.asarray(strata).astype(str)
    levels = levels if levels is not None else sorted(set(strata))
    T = V = 0.0
    for lev in levels:
        m = strata == lev
        n = int(m.sum()); n1 = int((m & is_case).sum()); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(values[m]); W = r[is_case[m]].sum(); EW = n1 * (n + 1) / 2
        _, cnt = np.unique(values[m], return_counts=True); tie = (cnt ** 3 - cnt).sum()
        VW = n1 * n0 / 12 * ((n + 1) - tie / (n * (n - 1)))
        T += (W - EW) / (n + 1); V += VW / (n + 1) ** 2
    z = T / np.sqrt(V) if V > 0 else 0.0
    return z, 2 * norm.sf(abs(z))


def batch_ratios(values, is_case, strata, levels):
    """Median case/control ratio within each stratum (nan when a stratum lacks one of the two groups)."""
    values = np.asarray(values, float); is_case = np.asarray(is_case, bool); strata = np.asarray(strata).astype(str)
    out = []
    for lev in levels:
        m = strata == lev
        c = np.median(values[m & ~is_case]) if (m & ~is_case).any() else 0
        out.append(float(np.median(values[m & is_case]) / c) if c > 0 and (m & is_case).any() else np.nan)
    return out


def mwu(a, b):
    """Two-sided Mann-Whitney p-value; nan when a group has fewer than three values."""
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) >= 3 and len(b) >= 3 else np.nan


TEXT_COLUMNS = {"batch": str, "sample_id": str, "date": str}


def cached(path, force=None):
    """Per-sample table left by an earlier raw pass, or None. Identifier columns are read as text: a batch
    label such as 20220215 would otherwise become an integer and no longer match the batch column.
    MINUT_RAW_FORCE=1 ignores the cache and forces a new pass."""
    force = os.environ.get("MINUT_RAW_FORCE", "0") == "1" if force is None else force
    path = Path(path)
    if path.exists() and not force:
        return pd.read_csv(path, dtype=TEXT_COLUMNS)
    return None


def raw_available(sub):
    """True when MINUT_RAW holds the sub-folder (covid, milk, lung*, maf)."""
    return any(RAW.glob(sub)) if "*" in sub else (RAW / sub).exists()
