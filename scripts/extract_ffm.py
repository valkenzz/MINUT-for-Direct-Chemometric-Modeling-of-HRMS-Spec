"""pyOpenMS FeatureFinderMetabo (FFM) extractor + cross-sample feature alignment.

Reference peak-picker baseline at DEFAULT parameters (no per-dataset tuning).
For each sample: MassTraceDetection -> ElutionPeakDetection -> FeatureFindingMetabo,
then a simple grid alignment (per (m/z, RT) tile, MAX intensity across samples).

Usage:
    python extract_ffm.py --dataset <covid|lung|milk> --manifest <manifest.csv>

Paths are environment-driven:
    MINUT_FEATURES_DIR   output dir for <dataset>_ffm.csv   (default: outputs/features)
    MINUT_DATA_DIR       root that holds the raw files       (default: data)
Manifest columns: sample_name, file_path, label. `file_path` may be an absolute
path or just a file name resolved under MINUT_DATA_DIR.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyopenms as oms

OUT_DIR = Path(os.environ.get("MINUT_FEATURES_DIR", "outputs/features"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("MINUT_DATA_DIR", "data"))

def resolve_raw(file_ref):
    p = Path(str(file_ref))
    if p.exists():
        return p
    cand = DATA_DIR / p.name
    return cand if cand.exists() else (DATA_DIR / p)

# alignment tolerances (dataset-specific, generous defaults)
TOL = {
    "covid": dict(mz_ppm=10.0, rt_sec=15.0),
    "lung":  dict(mz_ppm=20.0, rt_sec=30.0),
    "milk":  dict(mz_ppm=10.0, rt_sec=15.0),
}

def load_to_mzml(path: Path) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.FileHandler().loadExperiment(str(path), exp)
    return exp

def run_ffm(exp: oms.MSExperiment) -> oms.FeatureMap:
    mtd = oms.MassTraceDetection()
    p = mtd.getDefaults()
    p.setValue("mass_error_ppm", 20.0)
    p.setValue("noise_threshold_int", 1000.0)
    mtd.setParameters(p)
    traces = []
    mtd.run(exp, traces, 0)
    epd = oms.ElutionPeakDetection()
    p = epd.getDefaults()
    p.setValue("chrom_peak_snr", 3.0)
    p.setValue("width_filtering", "fixed")
    epd.setParameters(p)
    split_traces = []
    epd.detectPeaks(traces, split_traces)
    final_traces = split_traces if len(split_traces) > 0 else traces
    ffm = oms.FeatureFindingMetabo()
    p = ffm.getDefaults()
    p.setValue("isotope_filtering_model", "none")
    p.setValue("remove_single_traces", "false")
    p.setValue("mz_scoring_by_elements", "false")
    p.setValue("report_convex_hulls", "false")
    ffm.setParameters(p)
    fmap = oms.FeatureMap()
    chromatograms = []
    ffm.run(final_traces, fmap, chromatograms)
    return fmap

def extract_features(path: Path):
    exp = load_to_mzml(path)
    fmap = run_ffm(exp)
    return [(f.getMZ(), f.getRT(), f.getIntensity()) for f in fmap]

def align_and_matrix(per_sample, tol):
    all_pts = []
    sample_names = list(per_sample.keys())
    for si, sn in enumerate(sample_names):
        for mz, rt, I in per_sample[sn]:
            all_pts.append((mz, rt, I, si))
    if not all_pts:
        return pd.DataFrame(index=sample_names)
    arr = np.array(all_pts, dtype=np.float64)
    mz_min = arr[:, 0].min()
    med_mz = np.median(arr[:, 0])
    mz_step = max(1e-4, 2 * tol["mz_ppm"] * med_mz / 1e6)
    rt_step = 2 * tol["rt_sec"]
    mz_idx = ((arr[:, 0] - mz_min) // mz_step).astype(np.int64)
    rt_min = arr[:, 1].min()
    rt_idx = ((arr[:, 1] - rt_min) // rt_step).astype(np.int64)
    keys = mz_idx * (rt_idx.max() + 2) + rt_idx
    uniq, inv = np.unique(keys, return_inverse=True)
    M = np.zeros((len(sample_names), len(uniq)), dtype=np.float32)
    for k in range(len(arr)):
        s = int(arr[k, 3]); f = int(inv[k])
        if arr[k, 2] > M[s, f]:
            M[s, f] = arr[k, 2]
    cols = [f"feat_{i}" for i in range(len(uniq))]
    return pd.DataFrame(M, index=sample_names, columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(TOL), required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest)
    per_sample = {}
    t0 = time.time()
    for i, row in man.iterrows():
        sn = row["sample_name"]; p = resolve_raw(row["file_path"])
        t1 = time.time()
        try:
            feats = extract_features(p)
        except Exception as e:
            print(f"  [{i+1:3d}] FAIL {sn}: {e}", file=sys.stderr)
            per_sample[sn] = []
            continue
        per_sample[sn] = feats
        print(f"  [{i+1:3d}/{len(man)}] {sn:30s} {time.time()-t1:6.1f}s  n_feats={len(feats):5d}", flush=True)
    print(f"\nAligning {sum(len(v) for v in per_sample.values())} features...")
    M = align_and_matrix(per_sample, TOL[args.dataset])
    M.insert(0, "sample_name", M.index.values)
    M["label"] = man.set_index("sample_name").loc[M["sample_name"], "label"].values
    out = OUT_DIR / f"{args.dataset}_ffm.csv"
    M.to_csv(out, index=False)
    print(f"  -> {M.shape[0]} samples x {M.shape[1]-2} aligned features  -> {out}")
    print(f"Total elapsed: {(time.time()-t0)/60:.2f} min")

if __name__ == "__main__":
    main()
