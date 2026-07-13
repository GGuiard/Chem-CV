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
import chemiscope
import ase.data

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

def plot_trj_1d(
    time: np.ndarray,
    cv: np.ndarray,
    color: np.ndarray | None = None,
    label: str = "CV",
    color_label: str = r"$\log(bias)$",
    bounds: tuple = (None, None),
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """1-D CV trajectory with optional colormap."""
    fig, ax = plt.subplots(layout="constrained")

    stride = max(1, int(len(cv)//max_nb_points))

    if color is not None:
        sc = ax.scatter(
            time[::stride], cv[::stride],
            c=color[::stride],
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    else:
        ax.scatter(
            time[::stride], cv[::stride],
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel(f"time [ps]")
    ax.set_ylabel(label)
    ax.set_ylim(bounds)
    return fig


def plot_trj_energy(
    time: np.ndarray,
    energy: np.ndarray,
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """Unbiased energy trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    stride = max(1, int(len(time)//max_nb_points))
    ax.scatter(
        time[::stride], energy[::stride]-np.min(energy[::stride]),
        s=scatter_size, linewidths=0,
    )
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("E [eV]")
    return fig


def plot_trj_bias(
    time: np.ndarray,
    logbias: np.ndarray,
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """Log of the OPES bias trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    stride = max(1, int(len(time)//max_nb_points))
    ax.scatter(
        time[::stride], -logbias[::stride],
        s=scatter_size, linewidths=0,
    )
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"$-\log(bias)$")
    return fig


def plot_trj_rct(time: np.ndarray, rct: np.ndarray) -> plt.Figure:
    """OPES reweighting factor c(t) trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, rct)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("OPES rct")
    return fig


def plot_trj_zed(time: np.ndarray, zed: np.ndarray) -> plt.Figure:
    """OPES partition-function estimate Z trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, zed)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("OPES zed")
    return fig


def plot_trj_n(time: np.ndarray, n_eff: np.ndarray, n_ker: np.ndarray) -> plt.Figure:
    """Effective sample count and kernel count trajectories."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, n_eff, label=r"$n_{eff}$")
    ax.plot(time, n_ker, label=r"$n_{ker}$")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("N")
    ax.legend()
    return fig


def plot_trj_emec(time: np.ndarray, emec: np.ndarray) -> plt.Figure:
    """Total energy trajectory with mean ± std band."""
    time = time[1:]
    emec = emec[1:]
    mean, std = np.mean(emec), np.std(emec)

    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(
        time,
        -std,
        std,
        alpha=0.20,
        linewidth=0,
        label=f"std = {std:.4g} eV",
    )
    ax.plot(time, emec-mean)
    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"$E_{mec}$ [eV]")
    ax.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain")
    return fig


def plot_trj_temperature(time: np.ndarray, temperature: np.ndarray) -> plt.Figure:
    """Temperature trajectory with mean ± std band."""
    mean, std = np.mean(temperature), np.std(temperature)

    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(
        time,
        mean - std,
        mean + std,
        alpha=0.20,
        linewidth=0,
        label=f"std = {std:.1f} K",
    )
    ax.plot(time, temperature)
    ax.axhline(
        mean,
        linestyle="--",
        label=f"mean = {mean:.1f} K",
    )

    ax.set_xlabel("time [ps]")
    ax.set_ylabel("T [K]")
    ax.legend()
    return fig


def plot_trj_2d(
    cv1: np.ndarray,
    cv2: np.ndarray,
    color: np.ndarray | None = None,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    color_label: str = r"$\log(bias)$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
    symmetric: bool = False,
) -> plt.Figure:
    """2-D scatter of two CVs with optional colormap."""
    fig, ax = plt.subplots(layout="constrained")

    stride = max(1, int(len(cv1)//max_nb_points))

    if color is not None:
        sc = ax.scatter(
            cv1[::stride], cv2[::stride],
            c=color[::stride],
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)
        
    else:
        ax.scatter(
            cv1[::stride], cv2[::stride],
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

def plot_density_1d(
    grid: np.ndarray,
    density: np.ndarray,
    label: str = "CV",
    bounds: tuple = (None, None),
) -> plt.Figure:
    """Normalised probability density along a 1-D CV."""
    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(grid, density, alpha=0.25, linewidth=0)
    ax.plot(grid, density)
    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)
    return fig


def plot_density_2d(
    grid: list[np.ndarray],
    density: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    density_min: float = 0.01,
    symmetric: bool = False,
) -> plt.Figure:
    """2-D probability density."""
    fig, ax = plt.subplots(layout="constrained")

    extent = [grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()]

    im = ax.imshow(
        np.ma.masked_where(density<=density_min, density).T,
        origin="lower",
        extent=extent,
        aspect="equal" if symmetric else "auto",
        cmap=cm_fessa,
        interpolation="bicubic",
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability density")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    return fig


# -----------------------------------------------------------------------
# Free Energy Surface
# -----------------------------------------------------------------------

def plot_fes_1d(
    grid: np.ndarray,
    fes: np.ndarray,
    err: np.ndarray | None,
    label: str = "CV",
    bounds: tuple = (None, None),
    fes_max: float | None = None,
) -> plt.Figure:
    """1-D free-energy surface with optional error band."""
    fig, ax = plt.subplots(layout="constrained")

    if isinstance(err, np.ndarray):
        ax.fill_between(
            grid,
            fes - err,
            fes + err,
            alpha=0.25,
            linewidth=0,
        )

    ax.plot(grid, fes)
    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel(f"FES [eV]")
    ax.set_ylim(np.min(fes), fes_max)
    return fig


def plot_fes_2d(
    grid: list[np.ndarray],
    fes: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    fes_max: float | None = None,
    nb_levels: int = 11,
    symmetric: bool = False,
) -> plt.Figure:
    """2-D free-energy surface."""
    fig, ax = plt.subplots(layout="constrained")

    if fes_max is None: fes_max = np.max(fes)

    levels = np.linspace(np.min(fes), fes_max, nb_levels)

    cf = ax.contourf(
        grid[0],
        grid[1],
        np.ma.masked_where(fes>=fes_max, fes).T,
        levels,
        cmap=cm_fessa,
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(f"FES [eV]")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    if symmetric:
        ax.set_aspect("equal", "box")
    return fig


def plot_fes_err_2d(
    grid: list[np.ndarray],
    err: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    symmetric: bool = False,
) -> plt.Figure:
    """2-D FES error."""
    fig, ax = plt.subplots(layout="constrained")

    extent = [grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()]

    im = ax.imshow(
        np.ma.masked_invalid(err).T,
        origin="lower",
        extent=extent,
        aspect="equal" if symmetric else "auto",
        cmap=cm_fessa,
        interpolation="bicubic",
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"FES error [eV]")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    if symmetric:
        ax.set_aspect("equal", "box")
    return fig


# -----------------------------------------------------------------------
# Bias and Energy plots
# -----------------------------------------------------------------------

def plot_cv_bias(
    cv: np.ndarray,
    bias: np.ndarray,
    color: np.ndarray | None = None,
    cv_label: str = "CV",
    color_label: str = "time [ps]",
    cv_bounds: tuple = (None, None),
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """Scatter of the bias in function of the CV."""
    fig, ax = plt.subplots(layout="constrained")

    stride = max(1, int(len(cv)//max_nb_points))

    if color is not None:
        sc = ax.scatter(
            cv[::stride], -bias[::stride],
            c=color[::stride],
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    else:
        ax.scatter(
            cv[::stride], -bias[::stride],
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel(cv_label)
    ax.set_ylabel(r"$-\log(bias)$")
    ax.set_xlim(cv_bounds)
    return fig


def plot_cv_energy(
    cv: np.ndarray,
    energy: np.ndarray,
    color: np.ndarray | None = None,
    cv_label: str = "CV",
    color_label: str = "time [ps]",
    cv_bounds: tuple = (None, None),
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """Scatter of the energy in function of the CV."""
    fig, ax = plt.subplots(layout="constrained")

    stride = max(1, int(len(cv)//max_nb_points))

    if color is not None:
        sc = ax.scatter(
            cv[::stride], energy[::stride]-np.min(energy[::stride]),
            c=color[::stride],
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    else:
        ax.scatter(
            cv[::stride], energy[::stride]-np.min(energy[::stride]),
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel(cv_label)
    ax.set_ylabel("E [eV]")
    ax.set_xlim(cv_bounds)
    return fig


def plot_energy_bias(
    energy: np.ndarray,
    bias: np.ndarray,
    color: np.ndarray | None = None,
    color_label: str = "time [ps]",
    scatter_size: float = 4.0,
    max_nb_points: int = 10_000,
) -> plt.Figure:
    """Correlation between the energy and the bias"""
    fig, ax = plt.subplots(layout="constrained")

    stride = max(1, int(len(energy)//max_nb_points))

    if color is not None:
        sc = ax.scatter(
            energy[::stride]-np.min(energy[::stride]),
            -bias[::stride],
            c=color[::stride],
            s=scatter_size,
            cmap=cm_fessa,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    else:
        ax.scatter(
            energy[::stride]-np.min(energy[::stride]),
            -bias[::stride],
            s=scatter_size,
            linewidths=0
        )

    ax.set_xlabel("E [eV]")
    ax.set_ylabel(r"$-\log(bias)$")
    return fig
 

# -----------------------------------------------------------------------
# Chemiscope
# -----------------------------------------------------------------------

def plot_chemiscope(
    structures,
    info,
    map_x: str = None,
    map_y: str = None,
    map_color: str = None,
    atom: str = None,
    # bond: str = None,
    # gradient: str = None,
    adapt_radius: bool = False,
    fps: int = 20,
    fname: str = "chemiscope",
):

    properties = {}
    for label, content in info.items():
        properties[label] = {"target": "structure", "values": content.to_numpy().T}

    environments = chemiscope.all_atomic_environments(structures)

    shapes = {}

    settings = {
        "target": "structure",
        "map": {
            "x": {"property": map_x},
            "y": {"property": map_y},
            "color": {"property": map_color}
        },
        "structure": [{
            "bonds": True,
            "spaceFilling": False,
            "keepOrientation": True,
            "playbackDelay": int(1000//fps),
        }],
    }

    if atom is not None:
        nb_atoms = len(structures[0])
        values = np.array([info[f"{atom}.{i}"].to_list() for i in range(nb_atoms)])
        properties["_atom"] = {"target": "atom", "values": values.ravel()}
        settings["structure"][0] += {
            "atomLabels": True,
            "color": {
                "property": "_atom",
                "palette": "bwr",
                "min":-1,
                "max":1,
            },
        }
        
    if adapt_radius:
        atom_radius = []
        for atoms in structures:
            for atom in atoms:
                atom_radius.append({"radius": ase.data.covalent_radii[ase.data.chemical_symbols.index(atom.symbol)]})
        shapes["selection"] = {"kind": "sphere", "parameters": {"atom": atom_radius}}
        settings["shapes"] = "selection"

    chemiscope.write_input(f"{fname}.json.gz", structures=structures, properties=properties, environments=environments, shapes=shapes, settings=settings)


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
            bounds[1] = max(max([max(arr) for arr in ref]), max([max(arr) for arr in pred]))
        
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
