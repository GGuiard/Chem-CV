"""
figures.py
==========
Publication-quality figure generators for molecular dynamics post-processing.

Design philosophy
-----------------
* Fessa palette.
* Large fonts (14 pt base) and sparse ticks (≤4 major ticks per axis) for
  clean, readable journal panels.
* ``constrained_layout`` on every figure — zero wasted whitespace.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlcolvar.utils.plot import cm_fessa
from chemiscope import write_input, all_atomic_environments
from ase.data import covalent_radii, chemical_symbols

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    # --- Font sizes ---
    "font.size":           14,
    "axes.labelsize":      16,
    "axes.titlesize":      15,
    "xtick.labelsize":     13,
    "ytick.labelsize":     13,
    "legend.fontsize":     13,
    "figure.titlesize":    16,
    # --- Figure ---
    "figure.dpi":          150,
    "figure.figsize":      (5.0, 4.0),
    # --- Lines ---
    "lines.linewidth":     1.8,
    # --- Ticks: outward, only on visible spines ---
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    5,
    "ytick.major.size":    5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    # --- Legend ---
    "legend.frameon":      False,
    # --- Saving ---
    "savefig.bbox":        "tight",
    "savefig.dpi":         300,
})


# -----------------------------------------------------------------------
# Trajectories
# -----------------------------------------------------------------------

def trj(
    time: np.ndarray,
    cv: np.ndarray,
    label: str = "CV",
    color_value: np.ndarray | None = None,
    color_label: str = "bias [eV]",
    bounds: tuple = (None, None),
    time_unit: str | None = "ps",
    scatter_size: float = 4.0,
) -> plt.Figure:
    """
    1-D CV trajectory with optional colormap.
    """
    fig, ax = plt.subplots(layout="constrained")

    if color_value is not None:
        sc = ax.scatter(
            time, cv,
            c=color_value,
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    else:
        ax.scatter(
            time, cv,
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel(f"time [{time_unit}]" if time_unit else "time")
    ax.set_ylabel(label)
    ax.set_ylim(bounds)
    return fig


def trj_rct(time: np.ndarray, rct: np.ndarray) -> plt.Figure:
    """OPES reweighting factor c(t) trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, rct)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"OPES rct")
    return fig


def trj_zed(time: np.ndarray, zed: np.ndarray) -> plt.Figure:
    """OPES partition-function estimate Z trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, zed)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"OPES zed")
    return fig


def trj_n(time: np.ndarray, n_eff: np.ndarray, n_ker: np.ndarray) -> plt.Figure:
    """Effective sample count and kernel count trajectories."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, n_eff, label=r"$n_{eff}$")
    ax.plot(time, n_ker, label=r"$n_{ker}$")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("N")
    ax.legend()
    return fig


def trj_energy(energy: np.ndarray) -> plt.Figure:
    """
    Mechanical energy trajectory with mean ± std band.

    The first frame is discarded (often an outlier after equilibration).
    Plain scalar notation prevents matplotlib from adding an axis offset.
    """
    energy = energy[1:]
    mean, std = np.mean(energy), np.std(energy)

    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(
        np.arange(len(energy)),
        mean - std,
        mean + std,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(energy)
    ax.axhline(
        mean,
        linestyle="--",
        label=f"mean = {mean:.4g} eV"
    )

    ax.set_xlabel("frame index")
    ax.set_ylabel(r"$E_{mec}$ [eV]")
    ax.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain")
    return fig


def trj_temperature(temperature: np.ndarray) -> plt.Figure:
    """Temperature trajectory with mean ± std band."""
    mean, std = np.mean(temperature), np.std(temperature)

    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(
        np.arange(len(temperature)),
        mean - std,
        mean + std,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(temperature)
    ax.axhline(
        mean,
        linestyle="--",
        label=f"mean = {mean:.1f} K"
    )

    ax.set_xlabel("frame index")
    ax.set_ylabel("T [K]")
    ax.legend()
    return fig


def trj_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    color_value: np.ndarray | None = None,
    color_label: str = "bias [eV]",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    scatter_size: float = 4.0,
    symmetric: bool = False,
) -> plt.Figure:
    """
    2-D scatter of two CVs with optional colormap.
    """
    fig, ax = plt.subplots(layout="constrained")

    if color_value is not None:
        sc = ax.scatter(
            cv1, cv2,
            c=color_value,
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)
        
    else:
        ax.scatter(
            cv1, cv2,
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)
    if symmetric:
        ax.set_aspect("equal", "box")
    return fig


# -----------------------------------------------------------------------
# Densities
# -----------------------------------------------------------------------

def density(
    grid: np.ndarray,
    density_values: np.ndarray,
    label: str = "CV",
    bounds: tuple = (None, None),
) -> plt.Figure:
    """Normalised probability density along a 1-D CV."""
    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(grid, density_values, alpha=0.25, linewidth=0)
    ax.plot(grid, density_values)
    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)
    return fig


def density_2d(
    grid: list[np.ndarray],
    density_values: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    density_min: float | None = None,
    density_max: float = 1.0,
    symmetric: bool = False,
) -> plt.Figure:
    """
    2-D probability density.
    """
    fig, ax = plt.subplots(layout="constrained")

    vmin = density_min if density_min is not None else 0.0
    extent = [grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()]

    im = ax.imshow(
        density_values.T,
        origin="lower",
        extent=extent,
        aspect="equal" if symmetric else "auto",
        cmap=cm_fessa,
        vmin=vmin,
        vmax=density_max,
        interpolation="bicubic",
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability density")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)
    return fig


# -----------------------------------------------------------------------
# Free Energy Surface
# -----------------------------------------------------------------------

def fes(
    grid: np.ndarray,
    fes_values: np.ndarray,
    fes_error: np.ndarray | None,
    label: str = "CV",
    bounds: tuple = (None, None),
    fes_max: float | None = None,
    fes_units: str = "eV",
) -> plt.Figure:
    """
    1-D free-energy surface with optional error band.
    """
    fig, ax = plt.subplots(layout="constrained")

    if isinstance(fes_error, np.ndarray):
        ax.fill_between(
            grid,
            fes_values - fes_error,
            fes_values + fes_error,
            alpha=0.25,
            linewidth=0,
            label=r"$\pm 1 \sigma$ (block)",
        )
        ax.legend()

    ax.plot(grid, fes_values)
    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel(f"FES [{fes_units}]")
    ax.set_ylim(bottom=0, top=fes_max)
    return fig


def fes_2d(
    grid: list[np.ndarray],
    fes_values: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    fes_max: float | None = None,
    nb_levels: int = 11,
    fes_units: str | None = "eV",
    symmetric: bool = False,
) -> plt.Figure:
    """2-D free-energy surface."""
    fig, ax = plt.subplots(layout="constrained")

    levels = np.linspace(0, fes_max, nb_levels) if fes_max else nb_levels
    im = ax.contourf(grid[0], grid[1], fes_values, levels, cmap=cm_fessa)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"FES [{fes_units}]" if fes_units else "FES")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)
    if symmetric:
        ax.set_aspect("equal", "box")
    return fig


def fes_error_2d(
    grid: list[np.ndarray],
    fes_error: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    error_min: float | None = None,
    error_max: float | None = None,
    fes_units: str = "eV",
    symmetric: bool = False,
) -> plt.Figure:
    """
    2-D FES error.
    """
    fig, ax = plt.subplots(layout="constrained")

    vmin = error_min if error_min is not None else 0.0
    vmax = error_max if error_max is not None else float(np.nanmax(fes_error))
    extent = [grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()]

    im = ax.imshow(
        fes_error.T,
        origin="lower",
        extent=extent,
        aspect="equal" if symmetric else "auto",
        cmap=cm_fessa,
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"FES error [{fes_units}]" if fes_units else "FES error")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)
    if symmetric:
        ax.set_aspect("equal", "box")
    return fig


# -----------------------------------------------------------------------
# Chemiscope
# -----------------------------------------------------------------------

#TODO: generalize
def chemiscope(
    structures,
    time,
    d1,
    d2,
    chemcv = {},
    color = None,
    adapt_radius: bool = False,
    save_directory: str = "",
    save_name: str = "chemiscope",
):
    properties = {"d1": {"target": "structure",
                        "values": d1,
                        "description": "Distance between the carbon atom and the first chlorin atom"},
                  "d2": {"target": "structure",
                        "values": d2,
                        "description": "Coordination between the carbon atom and the second chlorin atom"},
                  "time": {"target": "structure",
                           "values": time,
                           "description": "time [ps]"}}
    
    for name, value in chemcv.items():
        properties[name] = {"target": "structure", "values": value}

    settings = {"target": "structure",
                "map": {"x": {"property": "d1"},
                        "y": {"property": "d2"},
                        "color": {"property": "time"}},
                "structure": [{"bonds": True,
                               "spaceFilling": False,
                               "keepOrientation": True,
                               "playbackDelay": 200}]}

    if isinstance(color, (list, np.ndarray)):
        properties["color"] = {"target": "atom",
                               "values": color.ravel(),
                               "description": "charge [e]"}
        settings["structure"][0] = settings["structure"][0] | {"atomLabels": True,
                                                               "labelsProperty": "color",
                                                               "color": {"property": "color", "palette": "bwr", "min":-1, "max":1}}
        
    if adapt_radius:
        atom_radius = []
        for atoms in structures:
            for atom in atoms:
                atom_radius.append({"radius": covalent_radii[chemical_symbols.index(atom.symbol)]})
        shapes = {"selection": {"kind": "sphere", "parameters": {"atom": atom_radius}}}
        settings["shapes"] = "selection"
    else:
        shapes = {}

    environments = all_atomic_environments(structures)

    write_input(f"{save_directory}/{save_name}.json.gz", structures=structures, properties=properties, environments=environments, shapes=shapes, settings=settings)

# -----------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------

def violin_correlation(
    bins,
    dataset,
    color: np.ndarray | None = None,
    xlabel: str = r"$CV_1$",
    ylabel: str = r"$CV_2$",
    color_label: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(layout="constrained")

    positions = (bins[1:] + bins[:-1]) / 2
    widths = (bins[1:] - bins[:-1]) / 2

    if color is None:
        vp = ax.violinplot(
            dataset,
            positions=positions,
            widths=widths,
            showmeans=True,
            showextrema=True,
            showmedians=True,
        )

    else:
        vp = ax.violinplot(
            dataset,
            positions=positions,
            widths=widths,
            showmeans=True,
            showextrema=True,
        )
        norm = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
        for body, color in zip(vp['bodies'], cm_fessa(norm(color))):
            body.set_facecolor(color)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm_fessa), ax=ax)
        cbar.set_label(color_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig


# -----------------------------------------------------------------------
# Machine Learning Model
# -----------------------------------------------------------------------

def pred(
    ref: np.ndarray,
    pred: np.ndarray,
    labels: list[str] | None = None,
    bounds: tuple[float] = (None, None),
    scatter_size: float = 4.0,
) -> plt.Figure:
    fig, ax = plt.subplots(layout="constrained")

    if ref.shape != pred.shape:
        raise f"ref and pred must have the same shape. Found {ref.shape} and {pred.shape}."
    
    if ref.ndim>2:
        raise f"ref and pred dimension must be 1 or 2. Found {ref.ndim}."
    
    if ref.ndim==1:
        if bounds[0] is None:
            bounds[0] = min(min(ref), min(pred))
        if bounds[1] is None:
            bounds[0] = max(max(ref), max(pred))

        ax.plot(list(bounds), list(bounds), c='k', ls='--')
        ax.scatter(ref, pred, s=scatter_size)

    else:
        if bounds[0] is None:
            bounds[0] = min(min([min(arr) for arr in ref]), min([min(arr) for arr in pred]))
        if bounds[1] is None:
            bounds[0] = max(max([max(arr) for arr in ref]), max([max(arr) for arr in pred]))
        
        if labels is None:
            labels = [str(i) for i in range(len(ref))]
        elif len(labels) != len(ref):
            raise f"labels must have the same size as ref and pred. Found {len(labels)} and {len(ref)}."

        ax.plot(list(bounds), list(bounds), c='k', ls='--')
        for x, y, label in zip(ref, pred, labels):
            ax.scatter(x, y, label=label, s=scatter_size)

        ax.legend()
    
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    ax.set_aspect('equal', 'box')

    return fig
