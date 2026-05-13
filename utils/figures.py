"""
figures.py
==========
Figure generators for molecular dynamics post-processing.

Design conventions
------------------
* Constrained layout.
* Scatter-based trajectories coloured by simulation time (magma colormap)
  to reveal how densely sampled each region is.
* Smooth gradient rendering (imshow + bicubic interpolation) for 2-D density
  and FES-error maps instead of discrete contour levels.
* Clean colorbar ticks via MaxNLocator / AutoLocator.
* Energy axes use plain scalar notation (no matplotlib offset labels).
* Ångström symbol (Å) used directly in default labels.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoLocator
from mlcolvar.utils.plot import cm_fessa

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def _apply_base_style(ax: plt.Axes) -> None:
    """Apply shared cosmetic settings to any Axes object."""
    # ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())


def _colorbar(fig: plt.Figure, mappable, ax: plt.Axes, label: str) -> mpl.colorbar.Colorbar:
    """Add a colorbar with clean tick spacing."""
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label)
    cbar.locator = MaxNLocator(nbins=5, integer=False)
    cbar.update_ticks()
    return cbar


# -----------------------------------------------------------------------
# 1-D Trajectories
# -----------------------------------------------------------------------

def trj(
    time: np.ndarray,
    cv: np.ndarray,
    label: str = "CV",
    bounds: tuple = (None, None),
    time_unit: str | None = "ps",
) -> plt.Figure:
    """
    Scatter plot of a 1-D collective variable over time, coloured by time
    so that densely sampled regions (slow dynamics) stand out visually.

    Parameters
    ----------
    time      : Simulation time array.
    cv        : CV values array.
    label     : Y-axis label (use LaTeX notation).
    bounds    : (ymin, ymax) limits for the CV axis.
    time_unit : Unit string appended to the x-axis label, or None.
    """
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(time, cv, s=1)
    ax.set_xlabel(f"time [{time_unit}]" if time_unit else "time")
    ax.set_ylabel(label)
    ax.set_ylim(bounds)
    _apply_base_style(ax)
    return fig


def trj_rct(time: np.ndarray, rct: np.ndarray) -> plt.Figure:
    """OPES reweighting factor c(t) trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, rct)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"OPES rct")
    _apply_base_style(ax)
    return fig


def trj_zed(time: np.ndarray, zed: np.ndarray) -> plt.Figure:
    """OPES partition-function estimate Z trajectory."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, zed)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel(r"OPES zed")
    _apply_base_style(ax)
    return fig


def trj_n(
    time: np.ndarray,
    n_eff: np.ndarray,
    n_ker: np.ndarray,
) -> plt.Figure:
    """Effective sample count and kernel count trajectories."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(time, n_eff, label=r"$n_{eff}$")
    ax.plot(time, n_ker, label=r"$n_{ker}$")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("N")
    ax.legend(frameon=False)
    _apply_base_style(ax)
    return fig


def trj_energy(energy: np.ndarray) -> plt.Figure:
    """
    Mechanical energy trajectory with mean ± std band.
    The first frame is discarded (often an outlier after equilibration).
    Plain scalar notation is enforced to avoid matplotlib's offset labels.
    """
    energy = energy[1:]
    mean, std = np.mean(energy), np.std(energy)

    fig, ax = plt.subplots(layout="constrained")
    ax.axhspan(mean - std, mean + std, alpha=0.30, linewidth=0)
    ax.plot(energy)
    ax.axhline(mean, color="k", linestyle="--", label=f"mean = {mean:.4g} eV")

    ax.set_xlabel("frame index")
    ax.set_ylabel(r"$E_{mec}$ [eV]")
    ax.legend(frameon=False)

    # Suppress the offset / scientific-notation label on the y-axis.
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain")
    _apply_base_style(ax)
    return fig


def trj_temperature(temperature: np.ndarray) -> plt.Figure:
    """Temperature trajectory with mean ± std band."""
    mean, std = np.mean(temperature), np.std(temperature)

    fig, ax = plt.subplots(layout="constrained")
    ax.axhspan(mean - std, mean + std, alpha=0.30, linewidth=0)
    ax.plot(temperature)
    ax.axhline(mean, color="k", linestyle="--", label=f"mean = {mean:.1f} K")

    ax.set_xlabel("frame index")
    ax.set_ylabel("T [K]")
    ax.legend(frameon=False)
    _apply_base_style(ax)
    return fig


# -----------------------------------------------------------------------
# 2-D Trajectories
# -----------------------------------------------------------------------

def trj_2d(
    time: np.ndarray,
    cv1: np.ndarray,
    cv2: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    time_unit: str | None = "ps",
    color_with_time: bool = True,
    scatter_size: float = 1,
    symmetric: bool = False,
) -> plt.Figure:
    """
    2-D scatter of two CVs, optionally coloured by simulation time.

    Colouring by time immediately reveals whether the trajectory visits the
    transition region frequently (dense colours) or only skims through it.
    """
    fig, ax = plt.subplots(layout="constrained")

    if color_with_time:
        sc = ax.scatter(cv1, cv2, c=time, s=scatter_size, cmap=cm_fessa)
        _colorbar(fig, sc, ax, f"time [{time_unit}]" if time_unit else "time")
    else:
        ax.scatter(cv1, cv2, s=scatter_size, alpha=0.5)

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)

    if symmetric:
        ax.set_aspect("equal", "box")

    _apply_base_style(ax)
    return fig


# -----------------------------------------------------------------------
# 1-D Density
# -----------------------------------------------------------------------

def density(
    grid: np.ndarray,
    density_values: np.ndarray,
    label: str = "CV",
    bounds: tuple = (None, None),
) -> plt.Figure:
    """Normalised probability density along a 1-D CV."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(grid, density_values)
    ax.fill_between(grid, density_values, alpha=0.15)
    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel("Probability density")
    ax.set_ylim(bottom=0)
    _apply_base_style(ax)
    return fig


# -----------------------------------------------------------------------
# 2-D Density
# -----------------------------------------------------------------------

def density_2d(
    grid: list[np.ndarray],
    density_values: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    density_min: float | None = None,
    density_max: float = 1.0,
    nb_levels: int = 11,
    smooth: bool = True,
    symmetric: bool = False,
) -> plt.Figure:
    """
    2-D probability density map.

    When *smooth* is True the map is rendered as a continuous gradient
    (imshow + bicubic interpolation) rather than discrete contour levels —
    cleaner for publication figures.
    """
    fig, ax = plt.subplots(layout="constrained")

    vmin = density_min if density_min is not None else 0.0

    if smooth:
        extent = [
            grid[0].min(), grid[0].max(),
            grid[1].min(), grid[1].max(),
        ]
        im = ax.imshow(
            density_values.T,
            origin="lower",
            extent=extent,
            aspect="auto" if not symmetric else "equal",
            cmap=cm_fessa,
            vmin=vmin,
            vmax=density_max,
            interpolation="bicubic",
        )
    else:
        levels = np.linspace(vmin, density_max, nb_levels)
        im = ax.contourf(grid[0], grid[1], density_values, levels, cmap=cm_fessa)

    _colorbar(fig, im, ax, "Probability density")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)

    if symmetric:
        ax.set_aspect("equal", "box")

    _apply_base_style(ax)
    return fig


# -----------------------------------------------------------------------
# 1-D Free Energy Surface
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
    1-D free-energy surface with optional block-bootstrap error band.

    Parameters
    ----------
    fes_error : Per-grid-point standard deviation from block analysis, or None.
    """
    fig, ax = plt.subplots(layout="constrained")

    if isinstance(fes_error, np.ndarray):
        ax.fill_between(
            grid,
            fes_values - fes_error,
            fes_values + fes_error,
            alpha=0.25,
            label=r"$\pm 1 \sigma$ (blocks)",
        )
        ax.legend(frameon=False, fontsize="small")

    ax.plot(grid, fes_values)

    ax.set_xlabel(label)
    ax.set_xlim(bounds)
    ax.set_ylabel(f"FES [{fes_units}]")
    ax.set_ylim(bottom=0, top=fes_max)
    _apply_base_style(ax)
    return fig


# -----------------------------------------------------------------------
# 2-D Free Energy Surface
# -----------------------------------------------------------------------

def fes_2d(
    grid: list[np.ndarray],
    fes_values: np.ndarray,
    cv1_label: str = r"$CV_1$",
    cv2_label: str = r"$CV_2$",
    cv1_bounds: tuple = (None, None),
    cv2_bounds: tuple = (None, None),
    fes_max: float | None = None,
    nb_levels: int = 11,
    fes_units: str = "eV",
    symmetric: bool = False,
) -> plt.Figure:
    """2-D free-energy surface rendered as filled contours (cm_fessa colormap)."""
    fig, ax = plt.subplots(layout="constrained")

    levels = np.linspace(0, fes_max, nb_levels) if fes_max else nb_levels
    im = ax.contourf(grid[0], grid[1], fes_values, levels, cmap=cm_fessa)

    _colorbar(fig, im, ax, f"FES [{fes_units}]")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)

    if symmetric:
        ax.set_aspect("equal", "box")

    _apply_base_style(ax)
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
    2-D FES block-bootstrap error map rendered as a smooth gradient
    (imshow + bicubic interpolation) to avoid artefacts from discrete levels.
    """
    fig, ax = plt.subplots(layout="constrained")

    vmin = error_min if error_min is not None else 0.0
    vmax = error_max if error_max is not None else float(np.nanmax(fes_error))

    extent = [
        grid[0].min(), grid[0].max(),
        grid[1].min(), grid[1].max(),
    ]
    im = ax.imshow(
        fes_error.T,
        origin="lower",
        extent=extent,
        aspect="auto" if not symmetric else "equal",
        cmap=cm_fessa,
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )

    _colorbar(fig, im, ax, f"FES error [{fes_units}]")

    ax.set_xlabel(cv1_label)
    ax.set_ylabel(cv2_label)
    ax.set_xlim(cv1_bounds)
    ax.set_ylim(cv2_bounds)

    if symmetric:
        ax.set_aspect("equal", "box")

    _apply_base_style(ax)
    return fig
