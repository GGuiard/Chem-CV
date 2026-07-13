"""
Phase-space scatter with DEAL-selected structures
"""

import argparse
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import plumed
from mlcolvar.utils.plot import cm_fessa

def plot_phase_space(
    xlim             = [1.5, 6.0],
    ylim             = [1.5, 6.0],
    stride           = 100,
    path_exploration = "COLVAR",
    path_selection   = "deal_selected.xyz",
    outdir           = ".",
):
    path_exploration = Path(path_exploration)
    path_selection   = Path(path_selection)

    data = plumed.read_as_pandas(str(path_exploration)).iloc[::stride]
    d1   = data["d1"].to_numpy()
    d2   = data["d2"].to_numpy()
    bias = data["opes.bias"].to_numpy()

    deal_traj     = ase.io.read(str(path_selection), ":")
    deal_selected = np.array([int(atoms.info["original_frame"]) for atoms in deal_traj])

    fig, ax = plt.subplots()
    sc = ax.scatter(d1, d2, c=bias, s=4, cmap=cm_fessa, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log(\mathrm{bias})$")

    ax.scatter(d1[deal_selected], d2[deal_selected], marker="*", c="r", edgecolors="0.5", s=128)

    ax.set_xlabel(r"$d_1$ [Å]")
    ax.set_ylabel(r"$d_2$ [Å]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlim[0] == ylim[0] and xlim[1] == ylim[1]:
        ax.set_aspect("equal", "box")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outdir / "phase-space.png"))

def type_float_list(arg: str) -> list[float]:
    return map(float, str(arg).split(','))

def _parser():
    p = argparse.ArgumentParser(description="Plot MD phase space + DEAL selected structures",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--exploration", type=str, default="COLVAR", help="PLUMED COLVAR file")
    p.add_argument("--selection", type=str, default="deal_selected.xyz", help="XYZ selected structures")
    p.add_argument("--xlim", type=type_float_list, nargs=2, default=[1.5, 6.0], help="x-axis limits")
    p.add_argument("--ylim", type=type_float_list, nargs=2, default=[1.5, 6.0], help="y-axis limits")
    p.add_argument("--stride", type=int, default=100, help="Stride for background scatter")
    p.add_argument("--outdir", default=".", help="Output directory")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    plot_phase_space(
        xlim             = args.xlim,
        ylim             = args.ylim,
        stride           = args.stride,
        path_exploration = args.exploration,
        path_selection   = args.selection,
        outdir           = args.outdir,
    )
