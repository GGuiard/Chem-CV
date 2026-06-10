"""
postprocessing.py
=================
Orchestrates the full post-processing pipeline for a MD simulation:

  1. Load COLVAR (and optionally ENERGY) output files via plumed.
  2. Compute densities and free-energy surfaces (FES) with
     error estimates.
  3. Save 
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plumed
import ase.io
import chemiscope
import pandas as pd

from .analyze import *
from .figures import *
from .helpers import remove_com_traj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocessing(
    # --- Collective variables ---
    cv: dict = {},
    cv_1d: list[str] = [],
    cv_2d: list[tuple[str, str]] = [],
    # --- Trajectory plotting ---
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
    # --- KDE parameters ---
    num_samples: int = 1000,
    bandwidth: float = 0.02,
    nb_levels: int = 11,
    # --- FES / reweighting ---
    temperature: float = 300.0,
    transient: float = None,
    nb_bootstraps: int = 100,
    bootstrap_rng = None,
    # --- FES / Density display ---
    density_min: float = 0.01,
    fes_max: float | None = None,
    # --- Chemiscope ---
    remove_com: bool = False,
    traj_data_stride: int = 10,
    traj_selection: str = "::10",
    map_x: str | None = None,
    map_y: str | None = None,
    map_color: str | None = None,
    adapt_radius: bool = False,
    fps: int = 20,
    # --- General options ---
    energy_label: str = "ene",
    opes_label: str = "opes",
    traj_file: str = "traj_comp.traj",
    directory: str = '',
    save: bool = True,
    format: str = "svg",
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
        CV key pairs for 2-D 

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

    save : bool
        Write figures to disk (SVG for 1-D, PNG for 2-D).

    show : bool
        Call ``plt.show()`` at the end.

    symmetric : bool
        Force equal aspect ratio for 2-D plots.
    """

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    data = {}

    # COLVAR
    colvar_path = Path(directory+"/COLVAR")
    if colvar_path.exists():
        try:
            data_colvar = plumed.read_as_pandas(str(colvar_path))
        except:
            data_colvar = pd.read_csv(str(colvar_path))

        data["time"] = data_colvar["time"].to_numpy().T

        for cv_name, cv_meta in cv.items():
            if cv_name in data_colvar.columns:
                cv_meta["values"] = data_colvar[cv_name].to_numpy().T
                if "bounds" not in cv_meta:
                    cv_meta["bounds"] = (np.min(cv_meta["values"]), np.max(cv_meta["values"]))
            else:
                warnings.warn(f"CV '{cv_name}' not found in COLVAR — skipping.", stacklevel=2)


        # Energy
        if energy_label in data_colvar.columns:
            data["energy"] = data_colvar[energy_label].to_numpy().T


        # Weights
        if f"{opes_label}.bias" in data_colvar.columns:
            data["bias"] = data_colvar[f"{opes_label}.bias"].to_numpy().T


        # OPES diagnostics
        if f"{opes_label}.rct" in data_colvar.columns:
            data["rct"] = data_colvar[f"{opes_label}.rct"].to_numpy().T
        if f"{opes_label}.zed" in data_colvar.columns:
            data["zed"] = data_colvar[f"{opes_label}.zed"].to_numpy().T
        if f"{opes_label}.neff" in data_colvar.columns:
            data["neff"] = data_colvar[f"{opes_label}.neff"].to_numpy().T
        if f"{opes_label}.nker" in data_colvar.columns:
            data["nker"] = data_colvar[f"{opes_label}.nker"].to_numpy().T
        
    else:
        warnings.warn("COLVAR file not found in the current directory.")


    # Info
    energy_path = Path(directory+"/ENERGY")
    if energy_path.exists():
        try:
            data_info = plumed.read_as_pandas(str(energy_path))
        except:
            data_info = pd.read_csv(str(energy_path))

        if "time" in data_info.columns:
            data["time_info"] = data_info["time"].to_numpy().T
        if "Emec" in data_info.columns:
            data["e_mec"] = data_info["Emec"].to_numpy().T
        if "Temp" in data_info.columns:
            data["temp"] = data_info["Temp"].to_numpy().T

    
    # Structures
    traj_path = Path(directory+'/'+traj_file)
    if traj_path.exists():    
        data["structures"] = ase.io.read(traj_path, traj_selection)
        if remove_com:
            data["structures"] = remove_com_traj(data["structures"])


    # -----------------------------------------------------------------------
    # Trajectories
    # -----------------------------------------------------------------------

    for cv_key, cv_meta in cv.items():
        if "values" not in cv_meta:
            continue
        if "time" in data:
            fig = plot_trj_1d(
                data["time"], cv_meta["values"],
                color=data["bias"],
                label=cv_meta["label"],
                bounds=cv_meta["bounds"],
                scatter_size=scatter_size,
                max_nb_points=max_nb_points,
            )
        else:
            fig = plot_trj_1d(
                data["time"], cv_meta["values"],
                label=cv_meta["label"],
                bounds=cv_meta["bounds"],
                scatter_size=scatter_size,
                max_nb_points=max_nb_points,
            )
        if save:
            fig.savefig(f"{directory}/trj_{cv_key}.{format}")


    if "time" in data and "energy" in data:
        fig = plot_trj_energy(data["time"], data["energy"])
        if save: fig.savefig(f"{directory}/trj_energy.{format}")


    if "time" in data and "bias" in data:
        fig = plot_trj_bias(data["time"], data["bias"])
        if save: fig.savefig(f"{directory}/trj_bias.{format}")


    if all(key in data for key in ("time", "rct", "zed", "neff", "nker")):
        fig = plot_trj_rct(data["time"], data["rct"])
        if save: fig.savefig(f"{directory}/trj_rct.{format}")

        fig = plot_trj_zed(data["time"], data["zed"])
        if save: fig.savefig(f"{directory}/trj_zed.{format}")

        fig = plot_trj_n(data["time"], data["neff"], data["nker"])
        if save: fig.savefig(f"{directory}/trj_n.{format}")


    if "time_info" in data and "e_mec" in data:
        fig = plot_trj_emec(data["time_info"], data["e_mec"])
        if save: fig.savefig(f"{directory}/trj_emec.{format}")


    if "time_info" in data and "e_mec" in data:
        fig = plot_trj_temperature(data["time_info"], data["temp"])
        if save: fig.savefig(f"{directory}/trj_temperature.{format}")


    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        fig = plot_trj_2d(
            cv1=cv[cv1_key]["values"],
            cv2=cv[cv2_key]["values"],
            color=data["bias"],
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            scatter_size=scatter_size,
            max_nb_points=max_nb_points,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"{directory}/trj2d_{cv1_key}_{cv2_key}.{format}")


    # -----------------------------------------------------------------------
    # Bias and energy
    # -----------------------------------------------------------------------
    
    if "time" in data and "bias" in data:
        for cv_key in cv_1d:
            if "values" not in cv[cv_key]:
                continue
            fig = plot_cv_bias(
                cv[cv_key]["values"], data["bias"],
                color=data["time"],
                cv_label=cv[cv_key]["label"],
                color_label="time [ps]",
                cv_bounds=cv[cv_key]["bounds"],
                scatter_size=scatter_size,
                max_nb_points=max_nb_points,
            )
            if save:
                fig.savefig(f"{directory}/bias_{cv_key}.{format}")

    
    if "time" in data and "energy" in data:
        for cv_key in cv_1d:
            if "values" not in cv[cv_key]:
                continue
            fig = plot_cv_energy(
                cv[cv_key]["values"], data["energy"],
                color=data["time"],
                cv_label=cv[cv_key]["label"],
                color_label="time [ps]",
                cv_bounds=cv[cv_key]["bounds"],
                scatter_size=scatter_size,
                max_nb_points=max_nb_points,
            )
            if save:
                fig.savefig(f"{directory}/energy_{cv_key}.{format}")

    
    if all(key in data for key in ("energy", "bias", "time")):
        fig = plot_energy_bias(
            data["energy"], data["bias"],
            color=data["time"],
            color_label="time [ps]",
            scatter_size=scatter_size,
            max_nb_points=max_nb_points,
        )
        if save:
            fig.savefig(f"{directory}/energy_bias.{format}")


    # -----------------------------------------------------------------------
    # Densities
    # -----------------------------------------------------------------------
    
    density_1d = {}
    for cv_key in cv_1d:
        if "values" not in cv[cv_key]:
            continue
        grid, density_1d[cv_key] = compute_density_1d(
            cv=cv[cv_key]["values"],
            bounds=cv[cv_key]["bounds"],
            num_samples=num_samples,
            bandwidth=bandwidth,
        )
        fig = plot_density_1d(
            grid, density_1d[cv_key],
            label=cv[cv_key]["label"],
            bounds=cv[cv_key]["bounds"])
        if save:
            fig.savefig(f"{directory}/density_{cv_key}.{format}")

    
    density_2d = {}
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        grid, density_2d[(cv1_key, cv2_key)] = compute_density_2d(
            cv1=cv[cv1_key]["values"],
            cv2=cv[cv2_key]["values"],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            num_samples=num_samples,
            bandwidth=bandwidth,
        )
        fig = plot_density_2d(
            grid, density_2d[(cv1_key, cv2_key)],
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            density_min=density_min,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"{directory}/density2d_{cv1_key}_{cv2_key}.{format}")


    # -----------------------------------------------------------------------
    # FES
    # -----------------------------------------------------------------------

    if transient is None:
        if show:
            plt.show()
        transient = float(input("Choose transient in ps:"))
    if "time" in data:
        transient_idx = int(np.searchsorted(data["time"], transient))
    else:
        transient_idx = 0

    if "bias" not in data:
        data["bias"] = np.ones(len(data["time"]))


    for cv_key in cv_1d:
        if "values" not in cv[cv_key]:
            continue
        grid, fes_1d, err_1d = compute_fes_1d(
            cv=cv[cv_key]["values"][transient_idx:],
            bias=data["bias"][transient_idx:],
            bounds=cv[cv_key]["bounds"],
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
            nb_bootstraps=nb_bootstraps,
            bootstrap_rng=bootstrap_rng,
        )
        fes_1d = np.ma.masked_where(density_1d[cv_key]<0.01, fes_1d)
        err_1d = np.ma.masked_where(density_1d[cv_key]<0.01, err_1d)
        fig = plot_fes_1d(
            grid, fes_1d, 25*err_1d,
            label=cv[cv_key]["label"],
            bounds=cv[cv_key]["bounds"],
            fes_max=fes_max,
        )
        if save:
            fig.savefig(f"{directory}/fes_{cv_key}.{format}")

    
    for cv1_key, cv2_key in cv_2d:
        if "values" not in cv[cv1_key] or "values" not in cv[cv2_key]:
            continue
        grid, fes_2d, err_2d = compute_fes_2d(
            cv1=cv[cv1_key]["values"][transient_idx:],
            cv2=cv[cv2_key]["values"][transient_idx:],
            bias=data["bias"][transient_idx:],
            cv1_bounds=cv[cv1_key]["bounds"],
            cv2_bounds=cv[cv2_key]["bounds"],
            temperature=temperature,
            num_samples=num_samples,
            bandwidth=bandwidth,
            nb_bootstraps=nb_bootstraps,
            bootstrap_rng=bootstrap_rng,
        )

        fes_2d = np.ma.masked_where(density_2d[(cv1_key, cv2_key)]<0.01, fes_2d)
        fig = plot_fes_2d(
            grid, fes_2d,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            fes_max=fes_max,
            nb_levels=nb_levels,
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"{directory}/fes2d_{cv1_key}_{cv2_key}.{format}")

        err_2d = np.ma.masked_where(density_2d[(cv1_key, cv2_key)]<0.01, err_2d)
        fig = plot_fes_err_2d(
            grid, err_2d,
            cv1_label=cv[cv1_key]["label"],
            cv2_label=cv[cv2_key]["label"],
            symmetric=symmetric,
        )
        if save:
            fig.savefig(f"{directory}/feserr2d_{cv1_key}_{cv2_key}.{format}")


    # -----------------------------------------------------------------------
    # Chemiscope
    # -----------------------------------------------------------------------

    start, end, stride = [int(n) if n!='' else None for n in traj_selection.split(':')]
    info = data_colvar.iloc[::traj_data_stride].iloc[start:end:stride]

    if (map_x is None or map_y is None) and len(cv_2d) > 0:
        map_x, map_y = cv_2d[0]
    if map_color is None and len(cv_1d) > 0:
        map_color = cv_1d[0]

    if "structures" in data:
        plot_chemiscope(
            data["structures"],
            info=info,
            map_x=map_x,
            map_y=map_y,
            map_color=map_color,
            adapt_radius=adapt_radius,
            fps=fps,
            fname=f"{directory}/chemiscope"
        )

    if show:
        plt.show()
        chemiscope.show_input(f"{directory}/chemiscope.json.gz")

