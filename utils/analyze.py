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
from scipy.stats import binned_statistic_2d
import ase.units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kbt_from_temp(temperature: float) -> float:
    """Return k_B T in eV for the given temperature in Kelvin."""
    return ase.units.kB * temperature


# def mask_within_bounds(
#     *arrays: np.ndarray,
#     bounds: tuple[float | None, float | None],
# ) -> np.ndarray:
#     """
#     Return a boolean index array selecting frames where *arrays[0]* lies
#     strictly within *bounds*.

#     All arrays are assumed to share the same length; the mask is built from
#     the first array only and should be applied to every array externally.

#     Parameters
#     ----------
#     *arrays : First array is used to compute the mask; rest are ignored here.
#     bounds  : (lower, upper) — either value may be None (no clipping).
#     """
#     x = arrays[0]
#     lower, upper = bounds
#     mask = np.ones(len(x), dtype=bool)
#     if lower is not None:
#         mask &= x > lower
#     if upper is not None:
#         mask &= x < upper
#     return mask


# def mask_2d_within_bounds(
#     cv1: np.ndarray,
#     cv2: np.ndarray,
#     cv1_bounds: tuple[float | None, float | None],
#     cv2_bounds: tuple[float | None, float | None],
# ) -> np.ndarray:
#     """Return a boolean mask selecting frames inside a 2-D bounding box."""
#     mask1 = mask_within_bounds(cv1, bounds=cv1_bounds)
#     mask2 = mask_within_bounds(cv2, bounds=cv2_bounds)
#     return mask1 & mask2


def bins_from_bounds(
    cv:np.ndarray[float],
    bounds: np.ndarray[float],
    num_samples: int
):
    lower, upper = bounds
    if lower is None:
        lower = np.min(cv)
    if upper is None:
        upper = np.max(cv)
    bins = np.linspace(lower, upper, num_samples)
    grid = (bins[1:] + bins[:-1]) / 2
    return bins, grid


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def compute_density(cv, bins):
    """
    Compute the density.
    Clip automatically cv values out of bins limits.

    Parameters
    ----------
    cv : 1D array (N,) for 1D, or (2, N) for 2D
    bins : 1D array for 1D, or [bins_x, bins_y] for 2D

    Returns
    -------
    density - is np.nan where bin was never visited
    """
    if cv.ndim == 1:
        density = np.histogram(cv, bins)[0]
        density = density.astype(float) / np.max(density)
        return density

    elif cv.ndim == 2:
        density = np.histogram2d(cv[0], cv[1], bins)[0]
        density = density.astype(float) / np.max(density)
        return density

    else:
        raise ValueError("cv must be shape (N,) or (2, N).")


def smooth_density(density, bandwidth):
    """
    Smooth the density by:
      1. convolving p with a Gaussian kernel (bandwidth in grid-bin units)
      2. masking unvisited bins (those that were nan before smoothing)

    Parameters
    ----------
    density : ndarray (1D or 2D), inf where bin was never visited
    bandwidth : float, smoothing width in grid-bin units

    Returns
    -------
    density_smooth : same shape, np.nan where density was nan, max shifted to 1
    """
    unvisited = np.where(density == 0)

    density_smooth = gaussian_filter(density.astype(float), sigma=bandwidth)

    density_smooth[unvisited] = np.nan

    finite = np.isfinite(density_smooth)
    density_smooth[finite] /= density_smooth[finite].max()

    return density_smooth


def compute_density_1d(
    cv: np.ndarray,
    bounds: tuple = (None, None),
    num_samples: int = 200,
    bandwidth: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an unweighted 1-D probability density via KDE.

    The density is derived from the KDE-based FES so that it is consistent
    with the weighted FES functions below.

    Returns
    -------
    grid    : 1-D array of CV grid points.
    density : Normalised probability density on the grid.
    """
    bins, grid = bins_from_bounds(cv, bounds, num_samples)

    density = compute_density(cv, bins=bins)

    if bandwidth != 0.0: density = smooth_density(density, bandwidth)

    return grid, density


def compute_density_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    num_samples: int = 200,
    bandwidth: float = 0.01,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Compute an unweighted 2-D probability density via KDE.

    Returns
    -------
    grid : [grid_cv1, grid_cv2] meshgrid arrays.
    density : 2-D normalised probability density.
    """
    bins_cv1, grid_cv1 = bins_from_bounds(cv1, cv1_bounds, num_samples)
    bins_cv2, grid_cv2 = bins_from_bounds(cv2, cv2_bounds, num_samples)
    bins, grid = [bins_cv1, bins_cv2], [grid_cv1, grid_cv2]

    density = compute_density(np.array([cv1, cv2]), bins)

    if bandwidth != 0.0: density = smooth_density(density, bandwidth)

    return grid, density


# ---------------------------------------------------------------------------
# FES
# ---------------------------------------------------------------------------

def compute_fes(cv, bias, bins, kbt):
    """
    Compute FES via log-sum-exp in log-weight space.
    Clip automatically cv values out of bins limits.

    Parameters
    ----------
    cv : 1D array (N,) for 1D, or (2, N) for 2D
    bias : 1D array (N,)
    bins : 1D array for 1D, or [bins_x, bins_y] for 2D
    kbt  : float

    Returns
    -------
    fes - is np.inf where bin was never visited
    """
    log_weights = bias / kbt
    log_weights -= np.max(log_weights)

    if cv.ndim == 1:
        bin_indices = np.digitize(cv, bins) - 1
        n_bins = len(bins) - 1

        # clip out-of-range indices
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        bi = bin_indices[valid]
        lw = log_weights[valid]

        log_pop = np.full(n_bins, -np.inf)
        order = np.argsort(bi)
        bi_sorted = bi[order]
        lw_sorted = lw[order]

        # reweight by bins
        splits = np.flatnonzero(np.diff(bi_sorted)) + 1
        for chunk_lw, bin_id in zip(
            np.split(lw_sorted, splits),
            bi_sorted[np.r_[0, splits]]
        ):
            log_pop[bin_id] = np.logaddexp.reduce(chunk_lw)

        fes = -kbt * log_pop
        finite = np.isfinite(fes)
        fes[finite] -= fes[finite].min()
        return fes

    elif cv.ndim == 2:
        nx, ny = len(bins[0]) - 1, len(bins[1]) - 1

        # the logaddexp accumulation preserves the log-space numerics
        _, _, _, binnumber = binned_statistic_2d(
            x = cv[0],
            y = cv[1],
            values=None,
            statistic='count',
            bins=bins,
            expand_binnumbers=True
        )
        # binnumber is (2, N)
        ix = binnumber[0] - 1
        iy = binnumber[1] - 1

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = ix[valid], iy[valid]
        lw = log_weights[valid]

        log_pop = np.full((nx, ny), -np.inf)
        flat_idx = ix * ny + iy

        order = np.argsort(flat_idx)
        flat_sorted = flat_idx[order]
        lw_sorted = lw[order]

        # reweight by bins
        splits = np.flatnonzero(np.diff(flat_sorted)) + 1
        for chunk_lw, fid in zip(
            np.split(lw_sorted, splits),
            flat_sorted[np.r_[0, splits]]
        ):
            i, j = divmod(fid, ny)
            log_pop[i, j] = np.logaddexp.reduce(chunk_lw)

        fes = -kbt * log_pop
        finite = np.isfinite(fes)
        fes[finite] -= fes[finite].min()
        return fes

    else:
        raise ValueError("cv must be shape (N,) or (2, N).")


def smooth_fes(fes, bandwidth, kbt):
    """
    Smooth the FES by:
      1. converting to probability: p = exp(-fes / kbt)
      2. convolving p with a Gaussian kernel (bandwidth in grid-bin units)
      3. converting back: fes_smooth = -kbt * log(p_smooth)
      4. masking unvisited bins (those that were inf/nan before smoothing)

    Smoothing in probability space preserves the logarithmic scale.

    Parameters
    ----------
    fes   : ndarray (1D or 2D), inf where bin was never visited
    bandwidth : float, smoothing width in grid-bin units
    kbt   : float

    Returns
    -------
    fes_smooth : same shape, np.inf where fes was inf, min shifted to 0
    """
    unvisited = ~np.isfinite(fes)

    prob = np.where(unvisited, 0.0, np.exp(-fes / kbt))

    prob_smooth = gaussian_filter(prob.astype(float), sigma=bandwidth)

    # Clip where probability is null
    prob_smooth = np.clip(prob_smooth, np.finfo(float).tiny, None)
    fes_smooth = -kbt * np.log(prob_smooth)

    fes_smooth[unvisited] = np.inf

    finite = np.isfinite(fes_smooth)
    fes_smooth[finite] -= fes_smooth[finite].min()

    return fes_smooth


def compute_fes_1d(
    cv: np.ndarray,
    bias: np.ndarray,
    bounds: tuple = (None, None),
    temperature: float = 300.0,
    num_samples: int = 200,
    bandwidth: float = 0.01,
    blocks: int = 3,
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
    bins, grid = bins_from_bounds(cv, bounds, num_samples)

    kbt = kbt_from_temp(temperature)

    rng = np.random.default_rng(bootstrap_rng)
    indices = rng.permutation(len(cv))
    bootstraps_indices = np.array_split(indices, blocks)

    bootstraps_fes = []
    for indices in bootstraps_indices:
        fes = compute_fes(cv[indices], bias[indices], bins, kbt)
        if bandwidth != 0.0: fes = smooth_fes(fes, bandwidth, kbt)
        bootstraps_fes.append(fes)

    fes = np.mean(bootstraps_fes, axis=0)
    err = np.std(bootstraps_fes, axis=0) / np.sqrt(blocks)

    return grid, fes, err


def compute_fes_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    bias: np.ndarray,
    cv1_bounds: tuple[float | None, float | None],
    cv2_bounds: tuple[float | None, float | None],
    temperature: float = 300.0,
    num_samples: int = 200,
    bandwidth: float = 0.01,
    blocks: int = 3,
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
    bins_cv1, grid_cv1 = bins_from_bounds(cv1, cv1_bounds, num_samples)
    bins_cv2, grid_cv2 = bins_from_bounds(cv2, cv2_bounds, num_samples)
    bins, grid = [bins_cv1, bins_cv2], [grid_cv1, grid_cv2]

    kbt = kbt_from_temp(temperature)

    rng = np.random.default_rng(bootstrap_rng)
    indices = rng.permutation(len(cv1))
    bootstraps_indices = np.array_split(indices, blocks)

    bootstraps_fes = []
    for indices in bootstraps_indices:
        fes = compute_fes(np.array([cv1[indices], cv2[indices]]), bias[indices], bins, kbt)
        if bandwidth != 0.0: fes = smooth_fes(fes, bandwidth, kbt)
        bootstraps_fes.append(fes)

    fes = np.mean(bootstraps_fes, axis=0)
    err = np.std(bootstraps_fes, axis=0) / np.sqrt(blocks)

    return grid, fes, err
