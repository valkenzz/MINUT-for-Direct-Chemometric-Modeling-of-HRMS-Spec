"""MINUT: max-intensity binning of an LC-HRMS run on a fixed (m/z, retention time) grid."""
from pathlib import Path

import numpy as np
from numba import njit

# (mz_min, mz_max, mz_width, rt_min, rt_max, rt_width); m/z in Th, RT in minutes.
GRIDS = {
    "milk": (100.0, 1000.0, 10.0, 2.0, 16.0, 2.0),
    "lung": (50.0, 1000.0, 2.0, 2.0, 12.0, 2.0),
    "covid": (100.0, 700.0, 0.5, 2.0, 20.0, 1.0),
}


def grid_shape(grid):
    """(n_rt, n_mz) bins; a trailing incomplete bin is dropped."""
    mz_min, mz_max, mz_width, rt_min, rt_max, rt_width = grid
    return int((rt_max - rt_min) / rt_width), int((mz_max - mz_min) / mz_width)


@njit
def _bin_max(rt, mz, intensity, mz_min, mz_max, mz_width, rt_min, rt_max, rt_width,
             max_i, max_mz, max_rt):
    n_rt, n_mz = max_i.shape
    inv_rt, inv_mz = 1.0 / rt_width, 1.0 / mz_width
    for k in range(intensity.size):
        if rt[k] < rt_min or rt[k] >= rt_max or mz[k] < mz_min or mz[k] >= mz_max:
            continue
        i = int((rt[k] - rt_min) * inv_rt)
        j = int((mz[k] - mz_min) * inv_mz)
        if i >= n_rt or j >= n_mz:
            continue
        # strict comparison: when two points have the same intensity the first one seen is kept
        if intensity[k] > max_i[i, j]:
            max_i[i, j] = intensity[k]
            max_mz[i, j] = mz[k]
            max_rt[i, j] = rt[k]


def minut(spectra, grid):
    """Bin a run and keep, per bin, its most intense point.

    `spectra` yields (rt_minutes, mz_array, intensity_array), one item per scan.
    Returns three (n_rt, n_mz) arrays: intensity, m/z and RT of the retained point.
    Empty bins have intensity 0 and nan coordinates.
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
        _bin_max(rt_array, mz, intensity, mz_min, mz_max, mz_width, rt_min, rt_max, rt_width,
                 max_i, max_mz, max_rt)
    return max_i, max_mz, max_rt


def feature_vector(max_i, max_mz, max_rt, grid):
    """Flatten to [intensity | m/z | RT], each block RT-major, all scaled to [0, 1].

    Intensity is divided by the run maximum; m/z and RT are min-max scaled with the grid
    limits so that every run shares the same coordinate frame. Empty bins are 0 everywhere.
    """
    mz_min, mz_max, _, rt_min, rt_max, _ = grid
    intensity = max_i / max_i.max() if max_i.max() > 0 else max_i
    mz = np.where(np.isnan(max_mz), mz_min, max_mz)
    rt = np.where(np.isnan(max_rt), rt_min, max_rt)
    return np.concatenate([
        intensity.ravel(),
        ((mz - mz_min) / (mz_max - mz_min)).ravel(),
        ((rt - rt_min) / (rt_max - rt_min)).ravel(),
    ])


def read_spectra(path, ms_levels=(1,)):
    """Yield (rt_minutes, mz, intensity) for the selected MS levels of an mzML or mzXML file.

    Use ms_levels=None to keep every scan. Retention times are returned in minutes.
    """
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


def transform_file(path, grid, ms_levels=(1,)):
    """MINUT feature vector of one run; see feature_vector for the layout."""
    max_i, max_mz, max_rt = minut(read_spectra(path, ms_levels), grid)
    return feature_vector(max_i, max_mz, max_rt, grid)
