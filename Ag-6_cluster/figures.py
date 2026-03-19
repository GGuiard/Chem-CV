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

def trj_r(time, r, transient=0):
    fig, ax = plt.subplots(layout='tight')
    if transient!=0:
        ax.axvspan(0, time[transient], color='grey', alpha=0.3)
    ax.plot(time, r, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel("Gyration")

    return fig

def trj_2D(c, r): # add color with time or make animation
    fig, ax = plt.subplots(layout='tight')
    ax.plot(c, r, 'o', ms=1)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("Gyration")

    return fig

def fes_c(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_r(grid, fes, err): # add pop and err
    fig, ax = plt.subplots(layout='tight')
    ax.fill_between(grid, fes-err, fes+err, alpha=0.3)
    ax.plot(grid, fes)

    ax.set_xlabel("Gyration")
    ax.set_ylabel("FES [eV]")
    
    return fig

def fes_2D(grid_c, grid_r, fes):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_c, grid_r, fes.T, 10, cmap=colormaps['Blues_r']) # cmo.tempo_r)
    ax.contour(grid_c, grid_r, fes.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("Gyration")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="FES [eV]")

    return fig

def err_fes_2D(grid_c, grid_r, err):
    fig, ax = plt.subplots(layout='tight')
    im = ax.contourf(grid_c, grid_r, err.T, 10, cmap=colormaps['Blues_r']) # cmo.tempo_r)
    ax.contour(grid_c, grid_r, err.T, 10, linestyles='-', colors='darkgray', linewidths=1.2)

    ax.set_xlabel("Coordination")
    ax.set_ylabel("Gyration")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label="Err(FES) [eV]")

    return fig

def av_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\langle Coordination \rangle$")

    return fig

def av_r(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\langle Gyration \rangle$")

    return fig

def delta_c(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\Delta Gyration$")

    return fig

def delta_r(time, av):
    fig, ax = plt.subplots(layout='tight')
    ax.plot(time, av, 'o', ms=1)

    ax.set_xlabel("t [fs]")
    ax.set_ylabel(r"$\Delta Gyration$")

    return fig
