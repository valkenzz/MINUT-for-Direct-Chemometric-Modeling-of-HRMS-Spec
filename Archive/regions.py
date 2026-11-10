"""Candidate-region statistics of the first revision, kept for the record.

These two tables were part of the first revision of the article and are no longer part of it: the raw-data
checks of the second revision (reproducibility/covid.ipynb, section 7, and reproducibility/milk.ipynb,
section 8) showed that bin 16044 holds three C18H37NO2 isomers that cannot be told apart, that the
valine-mass bins measure a background channel, and that the 138.05 ion of bin 183 is the oxonium fragment of
N-acetylhexosamine rather than PABA. The functions below reproduce the numbers as they were reported then,
from the deposited feature tables, with the loaders and bin tools of the repository root.

    python Archive/regions.py        # writes covid_regions.csv and milk_paba.csv next to this file
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from case_studies import GRIDS, bin_coordinates, load_covid, load_milk  # noqa: E402

HERE = Path(__file__).resolve().parent


def mass_cluster(mz, target, present, eps=0.01, min_samples=3):
    """Members of the DBSCAN m/z cluster that contains the point closest to the target mass."""
    labels = np.full(len(mz), -2)
    labels[present] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mz[present, None])
    nearest = np.flatnonzero(present)[np.argmin(np.abs(mz[present] - target))]
    return labels == labels[nearest]


def covid_regions():
    """Bins 16044 (then read as sphingosine d18:1 [M+Na]+) and 2436 / 1236 (then read as valine [M+H]+)."""
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
    """Bin 183 (m/z 130-140, RT 6-8 min): the 138.05 group, then read as PABA, by category and partition."""
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


if __name__ == "__main__":
    for name, run in (("covid_regions", covid_regions), ("milk_paba", milk_paba)):
        table = run()
        table.to_csv(HERE / f"{name}.csv", index=False, float_format="%.8g")
        print(f"\n{name}\n{table.to_string(index=False)}")
