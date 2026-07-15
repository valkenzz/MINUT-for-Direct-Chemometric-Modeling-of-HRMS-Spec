"""asari (default parameters) feature extractor.

Reference peak-picker baseline at DEFAULT parameters (no per-dataset tuning).
asari requires .mzML input; .mzXML is auto-converted via pyOpenMS.

Usage:
    python extract_asari.py --dataset <covid|lung|milk> --manifest <manifest.csv>

Environment-driven paths:
    MINUT_FEATURES_DIR   output dir for <dataset>_asari.csv  (default: outputs/features)
    MINUT_DATA_DIR       root that holds the raw files        (default: data)
    MINUT_WORK_DIR       scratch dir for asari staging/output (default: system temp)
Manifest columns: sample_name, file_path, label.
"""
from __future__ import annotations
import argparse, os, shutil, sys, time, glob, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path(os.environ.get("MINUT_FEATURES_DIR", "outputs/features"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("MINUT_DATA_DIR", "data"))
WORK = Path(os.environ.get("MINUT_WORK_DIR", tempfile.gettempdir())) / "asari_runs"
WORK.mkdir(parents=True, exist_ok=True)

def resolve_raw(file_ref):
    p = Path(str(file_ref))
    if p.exists():
        return p
    cand = DATA_DIR / p.name
    return cand if cand.exists() else (DATA_DIR / p)

def ensure_mzml(src: Path) -> Path:
    if src.suffix.lower() == ".mzml":
        return src
    dst = WORK / "converted" / (src.stem + ".mzML")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import pyopenms as oms
    exp = oms.MSExperiment()
    oms.FileHandler().loadExperiment(str(src), exp)
    oms.FileHandler().storeExperiment(str(dst), exp)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest)

    stage = WORK / f"{args.dataset}_in"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    file_map = {}
    print(f"Staging {len(man)} files into {stage}...", flush=True)
    for i, row in man.iterrows():
        src = ensure_mzml(resolve_raw(row["file_path"]))
        dst = stage / f"{row['sample_name']}.mzML"
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        file_map[row["sample_name"]] = dst.stem
    print(f"  staged: {len(file_map)} files", flush=True)

    out_dir = WORK / f"{args.dataset}_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    from asari.default_parameters import PARAMETERS
    from asari.main import process_project
    params = dict(PARAMETERS)
    params["outdir"] = str(out_dir)
    params["project_name"] = f"bench_{args.dataset}"
    params["multicores"] = 4
    params["mode"] = "pos"
    params["anno"] = False
    params["denovo"] = False
    params["min_prominence_threshold"] = int(0.33 * params["min_peak_height"])
    files = sorted(glob.glob(str(stage / "*.mzML")))
    print(f"Running asari on {len(files)} files with defaults...", flush=True)
    t0 = time.time()
    process_project(files, params)
    print(f"asari done in {(time.time()-t0)/60:.2f} min", flush=True)

    search_roots = [out_dir] + list(WORK.glob(f"{args.dataset}_out*"))
    candidates = []
    for r in search_roots:
        if r and r.exists():
            candidates += list(r.rglob("Feature_table.tsv"))
            candidates += list(r.rglob("full_Feature_table.tsv"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no Feature_table.tsv under {search_roots}")
    ft = pd.read_csv(candidates[0], sep="\t", low_memory=False)
    meta_cols = {"id_number", "mz", "rtime", "rtime_left_base", "rtime_right_base",
                 "parent_masstrack_id", "peak_area", "cSelectivity", "goodness_fitting",
                 "snr", "detection_counts"}
    sample_cols = [c for c in ft.columns if c not in meta_cols and not c.startswith("Unnamed")]
    keep = [c for c in sample_cols if c in file_map.values()]
    print(f"  features={len(ft)}  sample cols matched={len(keep)}/{len(sample_cols)}", flush=True)
    X = ft[keep].to_numpy(dtype=np.float32).T
    df = pd.DataFrame(X, index=list(keep), columns=[f"feat_{i}" for i in range(X.shape[1])])
    df.insert(0, "sample_name", df.index)
    df["label"] = df["sample_name"].map(dict(zip(man["sample_name"], man["label"]))).astype(int)
    out = OUT_DIR / f"{args.dataset}_asari.csv"
    df.to_csv(out, index=False)
    print(f"  -> {df.shape[0]} samples x {df.shape[1]-2} features  -> {out}", flush=True)
    print(f"Total elapsed: {(time.time()-t0)/60:.2f} min")

if __name__ == "__main__":
    main()
