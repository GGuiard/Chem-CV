import matplotlib.pyplot as plt
from matplotlib import colormaps
# import cmocean.cm as cmo
import numpy as np

# TODO: 
#   - adapt the plots to the system studied
#   - set aspect ratio 1:1 to phipsi plots
#   - give the possibility to not show err in fes1D plots
#   - setup save option, location, name, dpi, transparent... and an option to not show the plots if saved
#   - make multiple plots with all the trj, all the fes
#   - change cmap
#   - try scipy interpolation

def trj_E(Epot, Emec, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(Epot, label=r"$E_P$")
    ax.plot(Emec, label=r"$E_M$")
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("E [eV]")
    ax.legend()

    return fig

def trj_T(T, av, std):
    fig, ax = plt.subplots(layout='tight')
    ax.axhspan(av-std, av+std, color='grey', alpha=0.3)
    ax.plot(T)
    ax.axhline(av, color='k', linestyle='--')

    ax.set_xlabel("number of frame")
    ax.set_ylabel("T [K]")

    return fig

def trj_c(time, c, transient=0):
    fig, ax = plt.subplots(layout='tight')
    if transient!=0:
        ax.axvspan(0, time[transient], color='grey', alpha=0.3)
    ax.plot(time, c, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel("Coordination")

    return fig

def trj_d(time, d, transient=0):
    fig, ax = plt.subplots(layout='tight')
    if transient!=0:
        ax.axvspan(0, time[transient], color='grey', alpha=0.3)
    ax.plot(time, d, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$d_{N-N}\ [A]$")

    return fig

def trj_2D(c, d): # add color with time or make animation
    fig, ax = plt.subplots(layout='tight')
    ax.plot(c, d, 'o', ms=1)

    ax.set_xlabel("Coordination")
    ax.set_ylabel(r"$d_{N-N}\ [A]$")

    return fig

def fes_c(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_d(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel(r"$d_{N-N}\ [A]$")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_2D(grid_c, grid_d, fes):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_c, grid_d, fes.T, 10, cmap=colormaps['Blues_r']) # cmo.tempo_r)
    ax.contour(grid_c, grid_d, fes.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel("Coordination")
    ax.set_ylabel(r"$d_{N-N}\ [A]$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="FES [eV]")

    return fig

def err_fes_2D(grid_c, grid_d, err):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_c, grid_d, err.T, 10, cmap=colormaps['Blues_r']) # cmo.tempo_r)
    ax.contour(grid_c, grid_d, err.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel("Coordination")
    ax.set_ylabel(r"$d_{N-N} [A]$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Err(FES) [eV]")

    return fig

def av_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\langle Coordination \rangle$")

    return fig

def av_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\langle d_{N-N} \rangle\ [A]$")

    return fig

def delta_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\Delta Coordination$")

    return fig

def delta_d(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\Delta d_{N-N}\ [A]$")

    return fig
