"""
postprocessing.py
=================
Orchestrates the full post-processing pipeline for an OPES/metadynamics
MD simulation:

  1. Load COLVAR (and optionally ENERGY) output files via plumed.
  2. Compute densities and free-energy surfaces (FES) with block-bootstrap
     error estimates.
  3. Save publication-quality figures.

Usage
-----
Call `postprocessing(cv=..., ...)` from a per-system ``main.py`` that sets
the working directory and defines the CV metadata dictionary.
See ``main.py`` for a complete example.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plumed

import analyze
import figures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocessing(
    # --- Collective variables ---
    cv: dict | None = None,
    cv_1d: list[str] | None = None,
    cv_2d: list[tuple[str, str]] | None = None,
    # --- Trajectory plotting ---
    time_unit: str | None = "ps",
    color_with_time: bool = True,
    scatter_size: float = 0.8,
    # --- KDE parameters ---
    num_samples: int = 200,
    bandwidth: float = 0.01,
    nb_levels: int = 11,
    smooth_2d: bool = True,
    # --- FES / reweighting ---
    temperature: float = 300.0,
    transient: float = 0.0,
    blocks: int = 3,
    fes_units: str = "eV",
    # --- 1-D density & FES display limits ---
    fes_max_1d: float | None = None,
    # --- 2-D density display limits ---
    density_min_2d: float | None = None,
    # --- 2-D FES display limits ---
    fes_max_2d: float | None = None,
    # --- 2-D FES error display limits ---
    error_min_2d: float | None = None,
    error_max_2d: float | None = None,
    # --- General options ---
    save: bool = True,
    show: bool = True,
    symmetric: bool = False,
) -> None:
    """
    Run the full post-processing pipeline.

    Parameters
    ----------
    cv : dict
        Metadata for each collective variable, keyed by column name in COLVAR.
        Each entry is a dict with keys:

        * ``label``  - Axis label (LaTeX, e.g. ``r"$d_{C-Cl}$ [Å]"``).
        * ``bounds`` - ``(lower, upper)`` display and filter limits.

    cv_1d : list[str]
        Subset of CV keys for which 1-D FES figures are generated.
        Every CV also gets a trajectory and density figure.

    cv_2d : list[tuple[str, str]]
        Pairs of CV keys for which 2-D figures are generated.

    time_unit : str or None
        Label for time axes (e.g. ``"ps"``). Set to None to omit.

    color_with_time : bool
        Colour 2-D scatter trajectories by simulation time.

    num_samples : int
        Number of grid points for KDE estimation.

    bandwidth : float
        KDE bandwidth (in CV units).

    smooth_2d : bool
        Use smooth gradient rendering (imshow) for 2-D density and error
        maps instead of discrete contour levels.

    temperature : float
        Simulation temperature in Kelvin.

    transient : float
        Simulation time (in ps) to discard as transient for FES estimation.

    blocks : int
        Number of blocks for block-bootstrap error estimation.

    fes_units : str
        Energy unit for FES axes (passed through to mlcolvar).

    save : bool
        Write figures to disk as SVG (1-D) or PNG (2-D).

    show : bool
        Call ``plt.show()`` at the end.

    symmetric : bool
        Force equal aspect ratio for 2-D CV plots (useful when both axes
        share the same physical quantity).
    """
    if cv is None:
        cv = {}
    if cv_1d is None:
        cv_1d = []
    if cv_2d is None:
        cv_2d = []

    # -----------------------------------------------------------------------
    # Load COLVAR
    # -----------------------------------------------------------------------
    weights: np.ndarray | None = None
    rct = zed = n_eff = n_ker = None
    time: np.ndarray | None = None

    colvar_path = Path("COLVAR")
    if not colvar_path.exists():
        raise FileNotFoundError("COLVAR file not found in the current directory.")

    data = plumed.read_as_pandas(str(colvar_path))

    time = data["time"].to_numpy()
    # Determine the first frame index after the transient period.
    transient_idx = int(np.searchsorted(time, transient))

    for cv_name, cv_meta in cv.items():
        if cv_name in data.columns:
            cv_meta["values"] = data[cv_name].to_numpy()
        else:
            warnings.warn(f"CV '{cv_name}' not found in COLVAR — skipping.", stacklevel=2)

    # OPES-specific columns.
    if "opes.bias" in data.columns:
        kbt = analyze.kbt_from_temp(temperature)
        log_weights = data["opes.bias"].to_numpy()
        weights = np.exp(log_weights / kbt)

    opes_cols = {"opes.rct": None, "opes.zed": None, "opes.neff": None, "opes.nker": None}
    for col in opes_cols:
        if col in data.columns:
            opes_cols[col] = data[col].to_numpy()
    rct = opes_cols["opes.rct"]
    zed = opes_cols["opes.zed"]
    n_eff = opes_cols["opes.neff"]
    n_ker = opes_cols["opes.nker"]

    # -----------------------------------------------------------------------
    # Load ENERGY (optional)
    # -----------------------------------------------------------------------
    energy_path = Path("ENERGY")
    e_mec: np.ndarray | None = None
    temp_arr: np.ndarray | None = None

    if energy_path.exists():
        raw = np.loadtxt(str(energy_path))
        if raw.ndim == 2 and raw.shape[1] >= 2:
            e_mec, temp_arr = raw[:, 0], raw[:, 1]

    # -----------------------------------------------------------------------
    # 1-D Trajectory figures (all CVs)
    # -----------------------------------------------------------------------
    for cv_key, cv_meta in cv.items():
        if "values" not in cv_meta:
            continue
        fig = figures.trj(
            time,
            cv_meta["values"],
            label=cv_meta["label"],
            bounds=cv_meta["bounds"],
            time_unit=time_unit,
        )
        if save:
            fig.savefig(f"trj_{cv_key}.png", dpi=150)

    # OPES diagnostics.
    if all(arr is not None for arr in (rct, zed, n_eff, n_ker)):
        for name, func, arr in [
            ("rct", figures.trj_rct, rct),
            ("zed", figures.trj_zed, zed),
        ]:
            fig = func(time, arr)
            if save:
                fig.savefig(f"trj_{name}.svg", dpi=150)

        fig = figures.trj_n(time, n_eff, n_ker)
        if save:
            fig.savefig("trj_n.svg", dpi=150)

    if e_mec is not None:
        fig = figures.trj_energy(e_mec)
        if save:
            fig.savefig("trj_energy.svg", dpi=150)

    if temp_arr is not None:
        fig = figures.trj_temperature(temp_arr)
        if save:
            fig.savefig("trj_temperature.svg", dpi=150)

    # -----------------------------------------------------------------------
    # 2-D Trajectory figures
    # -----------------------------------------------------------------------
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        fig = figures.trj_2d(
            time,
            cv[cv1_key]["values"],
            cv[cv2_key]["values"],
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            time_unit=time_unit,
            color_with_time=color_with_time,
            scatter_size=scatter_size,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"trj2d_{cv1_key}_{cv2_key}.png", dpi=200)

    # -----------------------------------------------------------------------
    # 1-D Density figures (all CVs)
    # -----------------------------------------------------------------------
    for cv_key, cv_meta in cv.items():
        if "values" not in cv_meta:
            continue
        grid, dens = analyze.compute_density_1d(
            cv_meta["values"],
            bounds=cv_meta["bounds"],
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
        )
        fig = figures.density(
            grid,
            dens,
            label=cv_meta["label"],
            bounds=cv_meta["bounds"],
        )
        if save:
            fig.savefig(f"density_{cv_key}.svg", dpi=150)

    # -----------------------------------------------------------------------
    # 2-D Density figures
    # -----------------------------------------------------------------------
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        grid, dens = analyze.compute_density_2d(
            cv[cv1_key]["values"],
            cv[cv2_key]["values"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
        )
        fig = figures.density_2d(
            grid,
            dens,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            density_min=density_min_2d,
            smooth=smooth_2d,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"density2d_{cv1_key}_{cv2_key}.svg", dpi=150)

    # -----------------------------------------------------------------------
    # 1-D FES figures
    # -----------------------------------------------------------------------
    if weights is None:
        warnings.warn("No OPES bias found — 1-D FES will be unweighted.", stacklevel=2)

    for cv_key in cv_1d:
        if "values" not in cv[cv_key]:
            continue
        cv_post = cv[cv_key]["values"][transient_idx:]
        w_post = weights[transient_idx:] if weights is not None else np.ones(len(cv_post))

        grid, fes_vals, fes_err = analyze.compute_fes_1d(
            cv_post,
            bounds=cv[cv_key]["bounds"],
            weights=w_post,
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
            fes_units=fes_units,
            blocks=blocks,
        )
        fig = figures.fes(
            grid,
            fes_vals,
            fes_err,
            label=cv[cv_key]["label"],
            bounds=cv[cv_key]["bounds"],
            fes_max=fes_max_1d,
            fes_units=fes_units,
        )
        if save:
            fig.savefig(f"fes_{cv_key}.svg", dpi=150)

    # -----------------------------------------------------------------------
    # 2-D FES figures
    # -----------------------------------------------------------------------
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        cv1_post = cv[cv1_key]["values"][transient_idx:]
        cv2_post = cv[cv2_key]["values"][transient_idx:]
        w_post = weights[transient_idx:] if weights is not None else np.ones(len(cv1_post))

        grid, fes_vals, fes_err = analyze.compute_fes_2d(
            cv1_post,
            cv2_post,
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            weights=w_post,
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
            fes_units=fes_units,
            blocks=blocks,
        )

        fig = figures.fes_2d(
            grid,
            fes_vals,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            fes_max=fes_max_2d,
            nb_levels=nb_levels,
            fes_units=fes_units,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"fes2d_{cv1_key}_{cv2_key}.svg", dpi=150)

        fig = figures.fes_error_2d(
            grid,
            fes_err,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            error_min=error_min_2d,
            error_max=error_max_2d,
            fes_units=fes_units,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"feserr2d_{cv1_key}_{cv2_key}.svg", dpi=150)

    if show:
        plt.show()
