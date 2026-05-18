"""
postprocessing.py
=================
Orchestrates the full post-processing pipeline for a MD simulation:

  1. Load COLVAR (and optionally ENERGY) output files via plumed.
  2. Compute densities and free-energy surfaces (FES) with
     error estimates.
  3. Save figures.
"""

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
    scatter_size: float = 4.0,
    # --- KDE parameters ---
    num_samples: int = 200,
    bandwidth: float = 0.01,
    nb_levels: int = 11,
    # --- FES / reweighting ---
    temperature: float = 300.0,
    transient: float = 0.0,
    blocks: int = 3,
    fes_units: str = "eV",
    # --- 1-D FES display limits ---
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
        Each entry is a dict with:

        * ``label``  - Axis label (LaTeX, e.g. ``r"$d_{C-Cl}$ [Å]"``).
        * ``bounds`` - ``(lower, upper)`` display and filter limits.

    cv_1d : list[str]
        CV keys for 1-D FES figures (every CV gets trajectory + density).

    cv_2d : list[tuple[str, str]]
        CV key pairs for 2-D figures.

    time_unit : str or None
        Label for time axes (e.g. ``"ps"``).

    scatter_size : float
        Marker size (pts²) for trajectory scatter plots.

    num_samples : int
        KDE grid resolution.

    bandwidth : float
        KDE bandwidth in range scale unit.

    nb_levels : int
        Number of contour levels for 2-D FES.

    temperature : float
        Simulation temperature in Kelvin.

    transient : float
        Simulation time (ps) to discard before FES estimation.

    blocks : int
        Number of blocks for block-bootstrap error estimation.

    fes_units : str
        Energy unit string for FES axes.

    save : bool
        Write figures to disk (SVG for 1-D, PNG for 2-D).

    show : bool
        Call ``plt.show()`` at the end.

    symmetric : bool
        Force equal aspect ratio for 2-D plots.
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
    colvar_path = Path("COLVAR")
    if colvar_path.exists():
        data = plumed.read_as_pandas(str(colvar_path))
        time = data["time"].to_numpy()
        transient_idx = int(np.searchsorted(time, transient))

        for cv_name, cv_meta in cv.items():
            if cv_name in data.columns:
                cv_meta["values"] = data[cv_name].to_numpy()
            else:
                warnings.warn(f"CV '{cv_name}' not found in COLVAR — skipping.", stacklevel=2)

        # Weights
        if "opes.bias" in data.columns:
            kbt = analyze.kbt_from_temp(temperature)
            log_weights = data["opes.bias"].to_numpy()
            weights = np.exp(log_weights / kbt)

        # OPES diagnostics.
        opes = {col: data[col].to_numpy() if col in data.columns else None
                for col in ("opes.rct", "opes.zed", "opes.neff", "opes.nker")}
        rct, zed, n_eff, n_ker = (opes[k] for k in ("opes.rct", "opes.zed", "opes.neff", "opes.nker"))

    else:
        warnings.warn("COLVAR file not found in the current directory.")

    # -----------------------------------------------------------------------
    # Load ENERGY (optional)
    # -----------------------------------------------------------------------
    energy_path = Path("ENERGY")
    if energy_path.exists():
        e_mec, temp_arr = plumed.read_as_pandas(str(colvar_path)).to_numpy().T

    # -----------------------------------------------------------------------
    # 1-D Trajectory figures (all CVs)
    # -----------------------------------------------------------------------
    for cv_key, cv_meta in cv.items():
        if "values" not in cv_meta:
            continue
        fig = figures.trj(
            time,
            cv_meta["values"],
            log_weights=log_weights,
            label=cv_meta["label"],
            bounds=cv_meta["bounds"],
            time_unit=time_unit,
        )
        if save:
            fig.savefig(f"trj_{cv_key}.png")

    if all(arr is not None for arr in (rct, zed, n_eff, n_ker)):
        fig = figures.trj_rct(time, rct)
        if save: fig.savefig("trj_rct.svg")

        fig = figures.trj_zed(time, zed)
        if save: fig.savefig("trj_zed.svg")

        fig = figures.trj_n(time, n_eff, n_ker)
        if save: fig.savefig("trj_n.svg")

    if e_mec is not None:
        fig = figures.trj_energy(e_mec)
        if save: fig.savefig("trj_energy.svg")

    if temp_arr is not None:
        fig = figures.trj_temperature(temp_arr)
        if save: fig.savefig("trj_temperature.svg")

    # -----------------------------------------------------------------------
    # 2-D Trajectory figures
    # -----------------------------------------------------------------------
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        fig = figures.trj_2d(
            cv[cv1_key]["values"],
            cv[cv2_key]["values"],
            log_weights=log_weights,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            scatter_size=scatter_size,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"trj2d_{cv1_key}_{cv2_key}.png")

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
        fig = figures.density(grid, dens, label=cv_meta["label"], bounds=cv_meta["bounds"])
        if save:
            fig.savefig(f"density_{cv_key}.svg")

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
            grid, dens,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            density_min=density_min_2d,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"density2d_{cv1_key}_{cv2_key}.svg")

    # -----------------------------------------------------------------------
    # 1-D FES figures
    # -----------------------------------------------------------------------
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
            grid, fes_vals, fes_err,
            label=cv[cv_key]["label"],
            bounds=cv[cv_key]["bounds"],
            fes_max=fes_max_1d,
            fes_units=fes_units,
        )
        if save:
            fig.savefig(f"fes_{cv_key}.svg")

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
            cv1_post, cv2_post,
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
            grid, fes_vals,
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
            fig.savefig(f"fes2d_{cv1_key}_{cv2_key}.svg")

        fig = figures.fes_error_2d(
            grid, fes_err,
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
            fig.savefig(f"feserr2d_{cv1_key}_{cv2_key}.svg")

    if show:
        plt.show()
