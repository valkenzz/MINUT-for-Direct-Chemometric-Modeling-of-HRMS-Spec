#!/usr/bin/env python
"""Downstream benchmark evaluation for peak-picker feature tables.

Takes a samples x features matrix produced by a peak-picking tool (FFM/asari/XCMS)
plus the cohort split, and applies the identical leakage-free downstream pipeline
used for the MINUT rows:

  StandardScaler(with_mean=False) -> RandomForest(100 trees, random_state fixed), inside a
  Pipeline refit within every fold (no leakage). Cross-validation accuracy is the mean over
  a 10-fold stratified CV on the DISCOVERY (training) partition; test accuracy and
  AUC are computed on the VALIDATION partition.

Usage:
  python bench_eval.py --matrix <tool_matrix.csv> --split <manifest_with_cohort.csv>
                       [--name FFM]

Matrix columns: sample_name, feat_*, [label]
Split columns:  sample_name, file_path, cohort (discovery / validation), label
sample_name must match between matrix and split; if the matrix is keyed by raw-file basename
(XCMS output), it is mapped back through the split's file_path. Labels are taken from the split.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 0
N_TREES = 100
N_FOLDS = 10


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--name", default="tool")
    args = ap.parse_args()

    M = pd.read_csv(args.matrix)
    split = pd.read_csv(args.split)
    if "file_path" in split.columns:
        # some extractors (XCMS) key their rows by raw-file basename instead of the manifest sample name
        basenames = split["file_path"].astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
        if not M["sample_name"].isin(split["sample_name"]).any() and M["sample_name"].isin(basenames).any():
            M["sample_name"] = M["sample_name"].map(dict(zip(basenames, split["sample_name"])))
    keys = ["sample_name", "cohort"]
    if "label" in split.columns:
        # the manifest is the authority for labels; drop any (possibly empty) label column of the matrix
        M = M.drop(columns=["label"], errors="ignore")
        keys.append("label")
    M = M.merge(split[keys], on="sample_name", how="inner")
    if M.empty:
        raise SystemExit(f"no sample of {args.matrix} matches {args.split} (by sample_name or file basename)")

    feat_cols = [c for c in M.columns if c.startswith("feat_")]
    if not feat_cols:
        feat_cols = [c for c in M.columns
                     if c not in ("sample_name", "label", "cohort")]

    disc = M[M["cohort"] == "discovery"]
    val = M[M["cohort"] == "validation"]

    Xtr = disc[feat_cols].to_numpy(dtype=np.float64)
    ytr = disc["label"].to_numpy(dtype=int)
    Xte = val[feat_cols].to_numpy(dtype=np.float64)
    yte = val["label"].to_numpy(dtype=int)

    Xtr = np.nan_to_num(Xtr)
    Xte = np.nan_to_num(Xte)

    print(f"[{args.name}] features={len(feat_cols)}  "
          f"discovery={Xtr.shape[0]} (pos={ytr.sum()},neg={(ytr==0).sum()})  "
          f"validation={Xte.shape[0]} (pos={yte.sum()},neg={(yte==0).sum()})")

    # 10-fold stratified CV accuracy on the discovery partition (pipeline refit per fold)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(build_pipeline(), Xtr, ytr, cv=skf, scoring="accuracy")
    print(f"[{args.name}] CV accuracy (discovery, {N_FOLDS}-fold): "
          f"mean={cv.mean():.3f}  STD={cv.std():.3f}")

    # Fit on discovery, evaluate on validation
    pipe = build_pipeline().fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, proba)
    except ValueError:
        auc = float("nan")
    print(f"[{args.name}] TEST (validation): accuracy={acc:.3f}  AUC={auc:.3f}")


if __name__ == "__main__":
    main()
