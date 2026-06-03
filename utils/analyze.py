"""
analyze.py
==========
Statistical analysis utilities for molecular dynamics trajectories.

Provides helpers for:
  - Computing weighted free-energy surfaces (FES) via kernel-density
    estimation (KDE) with block-bootstrap error estimation.
  - Deriving normalised probability densities from FES values.
  - Filtering trajectory data to a rectangular CV region.

All compute routines return plain NumPy arrays so they can be used
independently of any particular plotting library.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import ase.units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kbt_from_temp(temperature: float) -> float:
    """Return k_B T in eV for the given temperature in Kelvin."""
    return ase.units.kB * temperature


def bounds_from_cv(
    cv: np.ndarray[float],
    bounds: tuple[float | None]
) -> tuple[float]:
    if bounds[0] is None:
        bounds[0] = np.min(cv)
    if bounds[1] is None:
        bounds[1] = np.max(cv)
    return bounds


def bins_from_bounds(
    bounds: tuple[float | None],
    num_samples: int
):
    bins = np.linspace(bounds[0], bounds[1], num_samples)
    grid = (bins[1:] + bins[:-1]) / 2
    return bins, grid


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def compute_density_1d(
    cv: np.ndarray,
    bounds: tuple = (None, None),
    num_samples: int = 1000,
    sigma: float = 20.,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an unweighted 1-D probability density via KDE.

    Returns
    -------
    grid    : 1-D array of CV grid points.
    density : Normalised probability density on the grid.
    """
    bounds = bounds_from_cv(cv, bounds)
    bins, grid = bins_from_bounds(bounds, num_samples)

    density = np.histogram(cv, bins)[0]
    if sigma != 0.0: density = gaussian_filter(density, sigma)
    density = density / np.max(density)

    return grid, density


def compute_density_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    num_samples: int = 1000,
    sigma: float = 20.,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Compute an unweighted 2-D probability density via KDE.

    Returns
    -------
    grid : [grid_cv1, grid_cv2] meshgrid arrays.
    density : 2-D normalised probability density.
    """
    cv1_bounds = bounds_from_cv(cv1, cv1_bounds)
    cv2_bounds = bounds_from_cv(cv2, cv2_bounds)
    bins_cv1, grid_cv1 = bins_from_bounds(cv1_bounds, num_samples)
    bins_cv2, grid_cv2 = bins_from_bounds(cv2_bounds, num_samples)
    bins, grid = [bins_cv1, bins_cv2], [grid_cv1, grid_cv2]

    density = np.histogram2d(cv1, cv2, bins)[0]
    if sigma != 0.0: density = gaussian_filter(density, sigma)
    density = density / np.max(density)

    return grid, density


# ---------------------------------------------------------------------------
# FES
# ---------------------------------------------------------------------------

def compute_fes_1d(
    cv: np.ndarray,
    bias: np.ndarray,
    bounds: tuple = (None, None),
    temperature: float = 300.0,
    num_samples: int = 1000,
    sigma: float = 20.,
    nb_bootstraps: int = 100,
    bootstrap_rng = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a reweighted 1-D FES with bootstrap error estimate.

    Parameters
    ----------
    cv          : CV trajectory array (post-transient slice expected).
    bounds      : CV domain (lower, upper).
    weights     : OPES/metadynamics reweighting factors (same length as cv).
    temperature : Temperature in Kelvin.
    num_samples : Number of KDE grid points.
    bandwidth   : KDE bandwidth.
    fes_units   : Output energy unit string passed to compute_fes.
    blocks      : Number of blocks for bootstrap error estimation.

    Returns
    -------
    grid   : 1-D grid array.
    fes    : Free-energy values on the grid.
    error  : Bootstrap standard deviation on the grid.
    """
    bounds = bounds_from_cv(cv, bounds)
    bins, grid = bins_from_bounds(bounds, num_samples)

    kbt = kbt_from_temp(temperature)

    log_weights  = bias / kbt
    log_weights -= log_weights.max()

    bin_indices = np.digitize(cv, bins) - 1
    n_bins      = len(bins) - 1
    log_pop     = np.full(n_bins, -np.inf)

    order     = np.argsort(bin_indices)
    bi_sorted = bin_indices[order]
    lw_sorted = log_weights[order]

    splits = np.flatnonzero(np.diff(bi_sorted)) + 1
    for chunk_lw, bin_id in zip(np.split(lw_sorted, splits),
                                bi_sorted[np.r_[0, splits]]):
        if 0 <= bin_id < n_bins:
            log_pop[bin_id] = np.logaddexp.reduce(chunk_lw)

    prob = np.exp(log_pop)

    if sigma != 0.0: prob = gaussian_filter(prob, sigma=sigma)
    fes = -kbt * np.ma.log(prob)
    fes -= np.min(fes[np.isfinite(fes)])

    if nb_bootstraps==0:
        return grid, fes

    rng = np.random.default_rng(bootstrap_rng)
    indices = rng.permutation(len(cv))
    bootstraps_indices = np.array_split(indices, nb_bootstraps)

    bootstraps_fes = []
    for indices in bootstraps_indices:

        log_weights  = bias[indices] / kbt
        log_weights -= log_weights.max()

        bin_indices = np.digitize(cv[indices], bins) - 1
        n_bins      = len(bins) - 1
        log_pop     = np.full(n_bins, -np.inf)

        order     = np.argsort(bin_indices)
        bi_sorted = bin_indices[order]
        lw_sorted = log_weights[order]

        splits = np.flatnonzero(np.diff(bi_sorted)) + 1
        for chunk_lw, bin_id in zip(np.split(lw_sorted, splits),
                                    bi_sorted[np.r_[0, splits]]):
            if 0 <= bin_id < n_bins:
                log_pop[bin_id] = np.logaddexp.reduce(chunk_lw)

        bootstrap_prob = np.exp(log_pop)

        # if sigma != 0.0: bootstrap_prob = gaussian_filter(bootstrap_prob, sigma=sigma)
        bootstrap_fes = -kbt * np.ma.log(bootstrap_prob)
        bootstrap_fes -= np.min(bootstrap_fes)
        bootstraps_fes.append(bootstrap_fes)

    err = np.std(bootstraps_fes, axis=0) / np.sqrt(nb_bootstraps)

    return grid, fes, err


def compute_fes_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bias: np.ndarray,
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    temperature: float = 300.0,
    num_samples: int = 1000,
    sigma: float = 20.,
    nb_bootstraps: int = 100,
    bootstrap_rng = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute a reweighted 2-D FES with bootstrap error estimate.

    Returns
    -------
    grid   : [grid_cv1, grid_cv2] meshgrid arrays.
    fes    : 2-D free-energy surface.
    error  : 2-D Bootstrap standard deviation.
    """
    cv1_bounds = bounds_from_cv(cv1, cv1_bounds)
    cv2_bounds = bounds_from_cv(cv2, cv2_bounds)
    bins_cv1, grid_cv1 = bins_from_bounds(cv1_bounds, num_samples)
    bins_cv2, grid_cv2 = bins_from_bounds(cv2_bounds, num_samples)
    bins, grid = [bins_cv1, bins_cv2], [grid_cv1, grid_cv2]

    kbt = kbt_from_temp(temperature)
    
    log_weights  = bias / kbt
    log_weights -= log_weights.max()

    bins_x, bins_y = bins
    nx, ny = len(bins_x) - 1, len(bins_y) - 1

    ix = np.digitize(cv1, bins_x) - 1
    iy = np.digitize(cv2, bins_y) - 1

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid], iy[valid]
    lw     = log_weights[valid]

    flat_idx = ix * ny + iy
    order       = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    lw_sorted   = lw[order]

    log_pop = np.full((nx, ny), -np.inf)
    splits  = np.flatnonzero(np.diff(flat_sorted)) + 1
    for chunk_lw, fid in zip(np.split(lw_sorted, splits),
                            flat_sorted[np.r_[0, splits]]):
        i, j = divmod(fid, ny)
        log_pop[i, j] = np.logaddexp.reduce(chunk_lw)

    prob = np.exp(log_pop)

    if sigma != 0.0: prob = gaussian_filter(prob, sigma=sigma)
    fes = -kbt * np.ma.log(prob)
    fes -= np.min(fes)

    if nb_bootstraps==0:
        return grid, fes

    rng = np.random.default_rng(bootstrap_rng)
    indices = rng.permutation(len(cv1))
    bootstraps_indices = np.array_split(indices, nb_bootstraps)

    bootstraps_fes = []
    for indices in bootstraps_indices:

        log_weights  = bias[indices] / kbt
        log_weights -= log_weights.max()

        bins_x, bins_y = bins
        nx, ny = len(bins_x) - 1, len(bins_y) - 1

        ix = np.digitize(cv1[indices], bins_x) - 1
        iy = np.digitize(cv2[indices], bins_y) - 1

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = ix[valid], iy[valid]
        lw     = log_weights[valid]

        flat_idx = ix * ny + iy
        order       = np.argsort(flat_idx)
        flat_sorted = flat_idx[order]
        lw_sorted   = lw[order]

        log_pop = np.full((nx, ny), -np.inf)
        splits  = np.flatnonzero(np.diff(flat_sorted)) + 1
        for chunk_lw, fid in zip(np.split(lw_sorted, splits),
                                flat_sorted[np.r_[0, splits]]):
            i, j = divmod(fid, ny)
            log_pop[i, j] = np.logaddexp.reduce(chunk_lw)

        bootstrap_prob = np.exp(log_pop)

        if sigma != 0.0: bootstrap_prob = gaussian_filter(bootstrap_prob, sigma=sigma)
        bootstrap_fes = -kbt * np.ma.log(bootstrap_prob)
        bootstrap_fes -= np.min(bootstrap_fes)
        bootstraps_fes.append(bootstrap_fes)

    err = np.std(bootstraps_fes, axis=0) / np.sqrt(nb_bootstraps)

    return grid, fes, err
