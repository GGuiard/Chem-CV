import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from chemiscope import write_input

def trj_E(Emec, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(Emec)
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("E [eV]")

    return fig

def trj_T(T, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(T)
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("T [K]")

    return fig

def trj_d(time, d, transient=0):
    fig, ax = plt.subplots(layout='tight')
    if transient!=0:
        ax.axvspan(0, time[transient], color='grey', alpha=0.3)
    ax.plot(time, d, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$d_{N-N}\ [A]$")

    return fig

def trj_c(time, c, transient=0):
    fig, ax = plt.subplots(layout='tight')
    if transient!=0:
        ax.axvspan(0, time[transient], color='grey', alpha=0.3)
    ax.plot(time, c, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("Coordination")

    return fig

def trj_z(time, z):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, z, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel("z [A]")

    return fig

def trj_2D(d, c): # add color with time or make animation
    fig, ax = plt.subplots(layout='tight')
    ax.plot(d, c, 'o', ms=1)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Coordination")

    return fig

def trj_xy(x, y): # add color with time or make animation
    fig, ax = plt.subplots(layout='tight')
    ax.plot(x, y, 'o', ms=1)

    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")

    return fig

def fes_d(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_c(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_2D(grid_d, grid_c, fes):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_c, fes.T, np.linspace(0, 1, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_c, fes.T, np.linspace(0, 1, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="FES [eV]")

    return fig

def err_fes_2D(grid_d, grid_c, err):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_d, grid_c, err.T, np.linspace(0, 0.02, 11), cmap=colormaps['Blues_r'])
    ax.contour(grid_d, grid_c, err.T, np.linspace(0, 0.02, 11), linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel(r"$d_{N-N} [A]$")
    ax.set_ylabel("Coordination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Err(FES) [eV]")

    return fig

def av_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\langle d_{N-N} \rangle\ [A]$")

    return fig

def av_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\langle Coordination \rangle$")

    return fig

def delta_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\Delta d_{N-N}\ [A]$")

    return fig

def delta_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$\Delta Coordination$")

    return fig

def chemiscope(traj, time, d, c):
    stride = len(time)//len(traj)+1
    properties = {"d": {"target": "structure",
                        "values": d[::stride],
                        "description": "Distance between the two atoms of nitrogen"},
                  "c": {"target": "structure",
                        "values": c[::stride],
                        "description": "Coordination between the atoms of nitrogen and the atoms of iron"},
                  "time": {"target": "structure",
                           "values": time[::stride],
                           "description": "time [ps]"}}
    
    write_input("chemiscope.json.gz", structures=traj, properties=properties)
