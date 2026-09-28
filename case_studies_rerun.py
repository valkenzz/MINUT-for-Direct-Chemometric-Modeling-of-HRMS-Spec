"""The three MINUT case studies of the paper: cow milk, lung cancer vs tuberculosis, COVID-19.

Everything starts from the deposited feature tables in data/ (one MINUT vector per run).
Running this file reruns the three classifications and the candidate-region statistics
and writes the numbers reported in the paper to results/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from minut import GRIDS, grid_shape

DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- cohorts

def load_milk():
    """177 raw-milk samples (organic = 1, conventional = 0), m/z and RT blocks only."""
    df = pd.read_pickle(DATA / "milk_10mz.pkl")
    df = df.drop_duplicates("echantillon")                           # first injection of each sample
    df = df[df["batch"].astype(str).isin(["20220215", "20220216"])]  # the two batches with both categories
    n_bins = int(np.prod(grid_shape(GRIDS["milk"])))
    X = np.stack(df["output_vector"].to_list())[:, n_bins:]          # intensity block dropped
    y = (df["categ"] == "b").to_numpy().astype(int)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    return X, y, train, test


def load_lung(mz_width=2.0):
    """236 serum samples of MTBLS6990 (cancer = 1, tuberculosis = 0), intensity and m/z blocks.

    The 85 discovery samples are the training set and the 151 validation samples the test
    set, as in the original study. Healthy controls, QC and blanks are not used. Tables for
    10, 2 and 0.5 m/z bins are deposited; the paper uses 2 m/z.
    """
    runs = pd.read_csv(DATA / "lung_injections.csv", dtype=str)
    runs = runs[runs["sample_type"] == "clinical"]
    df = pd.read_pickle(DATA / f"lung_{mz_width:g}mz_2min.pkl")
    df["sample_name"] = df["echantillon"].str.split(".").str[0]
    df = runs.merge(df, on="sample_name")
    grid = GRIDS["lung"][:2] + (mz_width,) + GRIDS["lung"][3:]
    n_bins = int(np.prod(grid_shape(grid)))
    X = np.stack(df["output_vector"].to_list())[:, :2 * n_bins]
    y = (df["disease"] == "lung cancer").to_numpy().astype(int)
    train = np.flatnonzero(df["cohort"] == "discovery")
    test = np.flatnonzero(df["cohort"] == "validation")
    return X, y, train, test


def load_covid():
    """63 plasma samples of MTBLS2291 (severe = 1, control = 0), one injection per subject."""
    runs = pd.read_csv(DATA / "covid_injections.csv", dtype=str)
    df = pd.read_pickle(DATA / "covid_0.5mz_1min.pkl")
    df["sample_name"] = df["echantillon"].str.split(".").str[0]
    df = df.merge(runs[["sample_name", "subject_id", "disease"]], on="sample_name")
    X = np.stack(df["output_vector"].to_list())
    y = (df["disease"] == "Severe COVID-19").to_numpy().astype(int)
    return X, y, df["subject_id"].to_numpy()


# --------------------------------------------------------------------------- feature selection

class GiniSelector(BaseEstimator, TransformerMixin):
    """Keep the k features with the largest random-forest Gini importance."""

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
    """COVID-19 selector: Gini pre-ranking, then mean of min-max scaled Mann-Whitney U and Fisher scores.

    U is computed with the controls as first sample, so the score favours features that are
    higher in controls than in severe cases.
    """

    def __init__(self, k=100, pre_rank=1000):
        self.k = k
        self.pre_rank = pre_rank

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=1000, random_state=0).fit(X, y)
        top = np.argsort(forest.feature_importances_)[::-1][:self.pre_rank]
        Xt = X[:, top]
        u = np.array([mannwhitneyu(Xt[y == 0, j], Xt[y == 1, j], alternative="two-sided").statistic
                      for j in range(Xt.shape[1])])
        fisher = (Xt[y == 1].mean(0) - Xt[y == 0].mean(0)) ** 2 / (Xt[y == 1].var(0) + Xt[y == 0].var(0) + 1e-8)
        scale = lambda v: MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel()
        score = (scale(u) + scale(fisher)) / 2
        order = sorted(range(len(top)), key=lambda j: score[j], reverse=True)   # stable: ties keep the Gini order
        self.indices_ = top[order[:self.k]]
        return self

    def transform(self, X):
        return X[:, self.indices_]


def threshold_selection(X, y, k, random_state):
    """Final-model selection of the original analysis: features whose importance reaches
    the (k+1)-th largest value. The threshold is inclusive, so k+1 features are kept
    (101 for milk, 7 for lung)."""
    forest = RandomForestClassifier(n_estimators=100, random_state=random_state).fit(X, y)
    importance = forest.feature_importances_
    return importance >= np.sort(importance)[::-1][k]


# --------------------------------------------------------------------------- classification

def holdout(X, y, train, test, keep, seed):
    """Fit on the training set with the selected features, score the held-out set."""
    model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X[train][:, keep], y[train])
    pred = model.predict(X[test][:, keep])
    proba = model.predict_proba(X[test][:, keep])[:, 1]
    boot = [accuracy_score(*resample(y[test], pred, random_state=i)) for i in range(1000)]
    return model, dict(
        test_accuracy=accuracy_score(y[test], pred), test_auc=roc_auc_score(y[test], proba),
        bootstrap_accuracy=np.mean(boot), bootstrap_sd=np.std(boot), n_features=int(keep.sum()),
    )


def milk():
    """Table 1, MINUT row."""
    X, y, train, test = load_milk()
    cv = cross_val_score(Pipeline([("select", GiniSelector(100, 0)),
                                   ("forest", RandomForestClassifier(n_estimators=100, random_state=2024))]),
                         X[train], y[train], cv=10)
    keep = threshold_selection(X[train], y[train], 100, random_state=0)
    model, scores = holdout(X, y, train, test, keep, seed=2024)
    # permutation test of the original analysis: label permutations of the test set,
    # 5-fold, on the already selected features
    _, _, p = permutation_test_score(model, X[test][:, keep], y[test], cv=5, n_permutations=100,
                                     scoring="accuracy", random_state=42)
    return dict(cv_accuracy=cv.mean(), cv_sd=cv.std(), permutation_p=p, **scores)


def lung():
    """Table 3 (MINUT row) and the lung cross-validation entry of Table 5."""
    X, y, train, test = load_lung()
    cv = cross_val_score(Pipeline([("select", GiniSelector(6, 2024)),
                                   ("forest", RandomForestClassifier(n_estimators=100, random_state=42))]),
                         X[train], y[train], cv=10)
    keep = threshold_selection(X[train], y[train], 6, random_state=2024)   # the "seven bins"
    _, scores = holdout(X, y, train, test, keep, seed=42)
    return dict(cv_accuracy=cv.mean(), cv_sd=cv.std(), **scores)


def covid():
    """Stratified 5-fold cross-validation on the 63 subjects, selection refitted in each fold."""
    X, y, _ = load_covid()
    pipeline = Pipeline([("select", HybridSelector(100)),
                         ("forest", RandomForestClassifier(n_estimators=100, random_state=0))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    accuracy, auc = [], []
    for train, test in folds.split(X, y):
        fitted = pipeline.fit(X[train], y[train])
        accuracy.append(accuracy_score(y[test], fitted.predict(X[test])))
        auc.append(roc_auc_score(y[test], fitted.predict_proba(X[test])[:, 1]))
    return dict(cv_accuracy=np.mean(accuracy), cv_sd=np.std(accuracy), cv_auc=np.mean(auc), auc_sd=np.std(auc),
                fold_auc=";".join(f"{v:.4f}" for v in auc))


# --------------------------------------------------------------------------- candidate regions

def bin_coordinates(X, bin_id, grid, components=("intensity", "mz", "rt")):
    """Physical intensity, m/z and RT of one bin for every sample of a feature matrix."""
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


def bin_window(bin_id, grid):
    """(m/z low, m/z high, RT low, RT high) of a bin index in the RT-major layout."""
    mz_min, _, mz_width, rt_min, _, rt_width = grid
    rt_bin, mz_bin = divmod(bin_id, grid_shape(grid)[1])
    return (mz_min + mz_bin * mz_width, mz_min + (mz_bin + 1) * mz_width,
            rt_min + rt_bin * rt_width, rt_min + (rt_bin + 1) * rt_width)


def mass_cluster(mz, target, present, eps=0.01, min_samples=3):
    """Members of the DBSCAN m/z cluster that contains the point closest to the target mass."""
    labels = np.full(len(mz), -2)
    labels[present] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mz[present, None])
    nearest = np.flatnonzero(present)[np.argmin(np.abs(mz[present] - target))]
    return labels == labels[nearest]


def covid_regions():
    """Figures 3-4 and the valine paragraph: sphingosine-compatible bin 16044, valine-compatible bins 2436 and 1236."""
    X, y, _ = load_covid()
    rows = []
    for bin_id, target, label in ((16044, 322.27165, "sphingosine d18:1 [M+Na]+"),
                                  (2436, 118.08626, "valine [M+H]+, RT 4-5 min"),
                                  (1236, 118.08626, "valine [M+H]+, RT 3-4 min")):
        coords = bin_coordinates(X, bin_id, GRIDS["covid"])
        keep = mass_cluster(coords["mz"].to_numpy(), target, coords["intensity"].to_numpy() > 0)
        control, severe = keep & (y == 0), keep & (y == 1)
        u, p = mannwhitneyu(coords["intensity"][control], coords["intensity"][severe], alternative="two-sided")
        mean_mz = coords["mz"][keep].mean()
        rows.append(dict(bin=bin_id, hypothesis=label, n=int(keep.sum()), n_control=int(control.sum()),
                         n_severe=int(severe.sum()), mean_mz=mean_mz,
                         sd_ppm=coords["mz"][keep].std(ddof=0) / mean_mz * 1e6,
                         mass_error_ppm=(mean_mz - target) / target * 1e6, U_control_first=u, p=p,
                         median_control=coords["intensity"][control].median(),
                         median_severe=coords["intensity"][severe].median()))
    return pd.DataFrame(rows)


def milk_paba():
    """Bin 183 (m/z 130-140, RT 6-8 min): the PABA-compatible region of Figure 2 and Figure S3."""
    X, y, train, test = load_milk()
    coords = bin_coordinates(X, 183, GRIDS["milk"], components=("mz", "rt"))
    in_mass = coords["mz"].between(138, 139).to_numpy()
    rows = []
    for scope, idx in (("all 177", np.arange(len(y))), ("training", train), ("test", test)):
        sel = in_mass[idx]
        conv, org = int((y[idx][sel] == 0).sum()), int((y[idx][sel] == 1).sum())
        table = [[conv, int((y[idx] == 0).sum()) - conv], [org, int((y[idx] == 1).sum()) - org]]
        rows.append(dict(scope=scope, n=int(sel.sum()), conventional=conv, organic=org,
                         fisher_p=fisher_exact(table)[1],
                         mean_mz=coords["mz"].to_numpy()[idx][sel].mean(), sd_mz=coords["mz"].to_numpy()[idx][sel].std(ddof=0),
                         mean_rt=coords["rt"].to_numpy()[idx][sel].mean(), sd_rt=coords["rt"].to_numpy()[idx][sel].std(ddof=0)))
    return pd.DataFrame(rows)


def importance_ranking(X, y, k, random_state=0, n_estimators=1000):
    """Indices of the k most important features of a forest fitted on (X, y)."""
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state).fit(X, y)
    return np.argsort(forest.feature_importances_)[::-1][:k]


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for name, run in (("milk", milk), ("lung", lung), ("covid", covid)):
        table = pd.DataFrame([run()])
        table.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.6f")
        print(f"\n{name}\n{table.T.to_string(header=False)}")
    for name, run in (("covid_regions", covid_regions), ("milk_paba", milk_paba)):
        table = run()
        table.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.8g")
        print(f"\n{name}\n{table.to_string(index=False)}")
