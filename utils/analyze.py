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

from __future__ import annotations

import numpy as np
from ase import units
from mlcolvar.utils.fes import compute_fes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kbt_from_temp(temperature: float) -> float:
    """Return k_B T in eV for the given temperature in Kelvin."""
    return units.kB * temperature


def mask_within_bounds(
    *arrays: np.ndarray,
    bounds: tuple[float | None, float | None],
) -> np.ndarray:
    """
    Return a boolean index array selecting frames where *arrays[0]* lies
    strictly within *bounds*.

    All arrays are assumed to share the same length; the mask is built from
    the first array only and should be applied to every array externally.

    Parameters
    ----------
    *arrays : First array is used to compute the mask; rest are ignored here.
    bounds  : (lower, upper) — either value may be None (no clipping).
    """
    x = arrays[0]
    lower, upper = bounds
    mask = np.ones(len(x), dtype=bool)
    if lower is not None:
        mask &= x > lower
    if upper is not None:
        mask &= x < upper
    return mask


def mask_2d_within_bounds(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_bounds: tuple[float | None, float | None],
    cv2_bounds: tuple[float | None, float | None],
) -> np.ndarray:
    """Return a boolean mask selecting frames inside a 2-D bounding box."""
    mask1 = mask_within_bounds(cv1, bounds=cv1_bounds)
    mask2 = mask_within_bounds(cv2, bounds=cv2_bounds)
    return mask1 & mask2


# ---------------------------------------------------------------------------
# 1-D analysis
# ---------------------------------------------------------------------------

def compute_density_1d(
    cv: np.ndarray,
    bounds: tuple[float | None, float | None],
    temperature: float = 300.0,
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
    kbt = kbt_from_temp(temperature)

    mask = mask_within_bounds(cv, bounds=bounds)
    cv_filtered = cv[mask]

    fes_values, grid, _, _ = compute_fes(
        cv_filtered,
        kbt=kbt,
        num_samples=num_samples,
        bounds=bounds,
        bandwidth=bandwidth,
    )

    density = np.exp(-fes_values / kbt)
    # Normalise so the maximum equals 1 (relative density).
    density /= density.max()

    return grid, density


def compute_fes_1d(
    cv: np.ndarray,
    bounds: tuple[float | None, float | None],
    weights: np.ndarray,
    temperature: float = 300.0,
    num_samples: int = 200,
    bandwidth: float = 0.01,
    fes_units: str = "eV",
    blocks: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a reweighted 1-D FES with block-bootstrap error estimate.

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
    error  : Block-bootstrap standard deviation on the grid.
    """
    mask = mask_within_bounds(cv, bounds=bounds)
    cv_filtered = cv[mask]
    weights_filtered = weights[mask]

    fes_values, grid, _, error = compute_fes(
        cv_filtered,
        temp=temperature,
        fes_units=fes_units,
        num_samples=num_samples,
        bounds=bounds,
        bandwidth=bandwidth,
        weights=weights_filtered,
        blocks=blocks,
    )

    return grid, fes_values, error


# ---------------------------------------------------------------------------
# 2-D analysis
# ---------------------------------------------------------------------------

def compute_density_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_bounds: tuple[float | None, float | None],
    cv2_bounds: tuple[float | None, float | None],
    temperature: float = 300.0,
    num_samples: int = 200,
    bandwidth: float = 0.01,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Compute an unweighted 2-D probability density via KDE.

    Returns
    -------
    grid    : [grid_cv1, grid_cv2] meshgrid arrays.
    density : 2-D normalised probability density.
    """
    kbt = kbt_from_temp(temperature)

    mask = mask_2d_within_bounds(cv1, cv2, cv1_bounds, cv2_bounds)
    X = np.stack((cv1[mask], cv2[mask])).T

    fes_values, grid, _, _ = compute_fes(
        X,
        fes_units="eV",
        kbt=kbt,
        num_samples=num_samples,
        bounds=[cv1_bounds, cv2_bounds],
        bandwidth=bandwidth,
    )

    density = np.exp(-fes_values / kbt)
    density /= density.max()

    return grid, density


def compute_fes_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_bounds: tuple[float | None, float | None],
    cv2_bounds: tuple[float | None, float | None],
    weights: np.ndarray,
    temperature: float = 300.0,
    num_samples: int = 200,
    bandwidth: float = 0.01,
    fes_units: str = "eV",
    blocks: int = 3,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute a reweighted 2-D FES with block-bootstrap error estimate.

    Returns
    -------
    grid   : [grid_cv1, grid_cv2] meshgrid arrays.
    fes    : 2-D free-energy surface.
    error  : 2-D block-bootstrap standard deviation.
    """
    mask = mask_2d_within_bounds(cv1, cv2, cv1_bounds, cv2_bounds)
    X = np.stack((cv1[mask], cv2[mask])).T
    weights_filtered = weights[mask]

    fes_values, grid, _, error = compute_fes(
        X,
        temp=temperature,
        fes_units=fes_units,
        num_samples=num_samples,
        bounds=[cv1_bounds, cv2_bounds],
        bandwidth=bandwidth,
        weights=weights_filtered,
        blocks=blocks,
    )

    return grid, fes_values, error
