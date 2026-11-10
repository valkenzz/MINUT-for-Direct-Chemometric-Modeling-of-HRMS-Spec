"""Command line: transform one mzML/mzXML run into a MINUT feature vector.

    python extract_minut.py run.mzXML --grid covid -o run.npz
    python extract_minut.py run.mzML --grid lung --mz-width 10 --all-ms-levels
    python extract_minut.py run.mzML --mz-range 50 1500 --rt-range 1 30 --mz-width 1 --rt-width 0.5

The grid is (mz_min, mz_max, mz_width, rt_min, rt_max, rt_width), RT in minutes. Start from one
of the three grids of the article (--grid) and override any of its parts, or give the ranges
and widths yourself. The .npz file holds the per-bin intensity, m/z and RT arrays, the grid
and the flattened feature vector used by the classifiers ([intensity | m/z | RT], see
minut.feature_vector).
"""
import argparse
from pathlib import Path

import numpy as np

from minut import GRIDS, feature_vector, grid_shape, minut, read_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="mzML or mzXML file")
    parser.add_argument("--grid", choices=GRIDS, default="covid", help="grid of one of the case studies (default: covid)")
    parser.add_argument("--mz-range", type=float, nargs=2, metavar=("MZ_MIN", "MZ_MAX"), help="m/z limits of the grid")
    parser.add_argument("--rt-range", type=float, nargs=2, metavar=("RT_MIN", "RT_MAX"), help="retention-time limits of the grid (minutes)")
    parser.add_argument("--mz-width", type=float, help="m/z bin width")
    parser.add_argument("--rt-width", type=float, help="RT bin width (minutes)")
    parser.add_argument("--normalization", choices=("max", "tic", "none"), default="max",
                        help="intensity block: per-sample maximum (article), total ion current, or raw counts")
    parser.add_argument("--all-ms-levels", action="store_true", help="bin every scan, not only MS1")
    parser.add_argument("-o", "--output", type=Path, help="destination .npz (default: next to the input)")
    args = parser.parse_args()

    mz_min, mz_max, mz_width, rt_min, rt_max, rt_width = GRIDS[args.grid]
    if args.mz_range:
        mz_min, mz_max = args.mz_range
    if args.rt_range:
        rt_min, rt_max = args.rt_range
    grid = (mz_min, mz_max, args.mz_width or mz_width, rt_min, rt_max, args.rt_width or rt_width)
    if not (mz_max > mz_min and rt_max > rt_min and grid[2] > 0 and grid[5] > 0):
        parser.error("the grid needs mz_max > mz_min, rt_max > rt_min and positive bin widths")
    if min(grid_shape(grid)) < 1:
        parser.error("the bins are wider than the grid")

    tic = 0.0

    def spectra():
        nonlocal tic
        for rt, mz, intensity in read_spectra(args.input, None if args.all_ms_levels else (1,)):
            tic += float(np.asarray(intensity, dtype=float).sum())
            yield rt, mz, intensity

    max_i, max_mz, max_rt = minut(spectra(), grid)
    vector = feature_vector(max_i, max_mz, max_rt, grid, normalization=args.normalization, tic=tic)
    output = args.output or args.input.with_suffix(".npz")
    np.savez_compressed(output, intensity=max_i, mz=max_mz, rt=max_rt, vector=vector, grid=np.array(grid),
                        normalization=args.normalization)
    n_rt, n_mz = max_i.shape
    print(f"{output}: m/z [{mz_min:g}, {mz_max:g}) by {grid[2]:g}, RT [{rt_min:g}, {rt_max:g}) min by {grid[5]:g} "
          f"-> {n_rt} x {n_mz} bins, {int((max_i > 0).sum())} non-empty; normalization: {args.normalization}")


if __name__ == "__main__":
    main()
