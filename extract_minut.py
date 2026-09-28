"""Command line: transform one mzML/mzXML run into a MINUT feature vector.

    python extract_minut.py run.mzXML --grid covid -o run.npz
    python extract_minut.py run.mzML --grid lung --mz-width 10 --all-ms-levels

The .npz file holds the per-bin intensity, m/z and RT arrays and the flattened
feature vector used by the classifiers ([intensity | m/z | RT], see minut.feature_vector).
"""
import argparse
from pathlib import Path

import numpy as np

from minut import GRIDS, feature_vector, minut, read_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="mzML or mzXML file")
    parser.add_argument("--grid", choices=GRIDS, default="covid", help="grid of one of the case studies")
    parser.add_argument("--mz-width", type=float, help="override the m/z bin width")
    parser.add_argument("--rt-width", type=float, help="override the RT bin width (minutes)")
    parser.add_argument("--all-ms-levels", action="store_true", help="bin every scan, not only MS1")
    parser.add_argument("-o", "--output", type=Path, help="destination .npz (default: next to the input)")
    args = parser.parse_args()

    mz_min, mz_max, mz_width, rt_min, rt_max, rt_width = GRIDS[args.grid]
    grid = (mz_min, mz_max, args.mz_width or mz_width, rt_min, rt_max, args.rt_width or rt_width)
    scans = read_spectra(args.input, None if args.all_ms_levels else (1,))
    max_i, max_mz, max_rt = minut(scans, grid)
    output = args.output or args.input.with_suffix(".npz")
    np.savez_compressed(output, intensity=max_i, mz=max_mz, rt=max_rt,
                        vector=feature_vector(max_i, max_mz, max_rt, grid), grid=np.array(grid))
    print(f"{output}: {max_i.shape[0]} x {max_i.shape[1]} bins, {int((max_i > 0).sum())} non-empty")


if __name__ == "__main__":
    main()
