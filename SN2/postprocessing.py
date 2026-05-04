import os
os.chdir("SN2/OPES_50")

import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.io import read
import plumed

import analyze
import figures
from orca_parser import ChemCV

# Simulation parameters
T = 300
kT = units.kB*T

# Postprocessing parameters
transient = 0
nb_bins_d = 100
sigma_d = 0.2
nb_bootstraps_1D, nb_bootstraps_2D = 10, 10
traj_start, traj_end, traj_stride = 0, 500000, 1

# Postprocessing options
use_energy = False
use_weights = True
use_opes = True
use_chemcv = False
use_traj = False

make_traj = True
make_density = True
make_fes = True

### Import data ###

if use_energy: Emec, Temp = np.loadtxt("ENERGY").T

data = plumed.read_as_pandas("COLVAR")
time, dd, d1, d2 = data[["time", "dd", "d1", "d2"]].to_numpy().T
if use_weights: weights = analyze.logw_to_w(data["opes.bias"].to_numpy().T, kT)
if use_opes: rct, zed, neff, nker = data[["opes.rct", "opes.zed", "opes.neff", "opes.nker"]].to_numpy().T

if use_chemcv:
    chemcv = ChemCV.load()

if use_traj: traj = read("traj.xyz", f"{traj_start}:{traj_end}:{traj_stride}")

### Postprocessing ###

if not use_weights: weights = time

if use_energy:
    av_Emec, std_Emec = np.average(Emec), np.std(Emec)
    av_Temp, std_Temp = np.average(Temp), np.std(Temp)

if make_density or make_fes:
    bins_dd, bins_d = np.linspace(-6, 6, nb_bins_d), np.linspace(0, 6, nb_bins_d)
    grid_dd, grid_d = analyze.bin_to_grid(bins_dd), analyze.bin_to_grid(bins_d)

if make_density:
    density_dd = analyze.population(dd, bins_dd, sigma_d)
    density_d1 = analyze.population(d1, bins_d, sigma_d)
    density_d2 = analyze.population(d2, bins_d, sigma_d)

    density_2D = analyze.population_2d(d1, d2, (bins_d, bins_d), (sigma_d, sigma_d))

if make_fes:
    pop_dd = analyze.population(dd[transient:], bins_dd, sigma_d, weights[transient:], use_weights)
    fes_dd = analyze.fes(pop_dd, kT)
    _, _, pop_list_dd = analyze.bootstrap_pop(dd[transient:], bins_dd, sigma_d, nb_bootstraps_1D, weights[transient:], use_weights)
    _, err_fes_dd, _ = analyze.error_fes(pop_list_dd, kT)

    pop_d1 = analyze.population(d1[transient:], bins_d, sigma_d, weights[transient:], use_weights)
    fes_d1 = analyze.fes(pop_d1, kT)
    _, _, pop_list_d1 = analyze.bootstrap_pop(d1[transient:], bins_d, sigma_d, nb_bootstraps_1D, weights[transient:], use_weights)
    _, err_fes_d1, _ = analyze.error_fes(pop_list_d1, kT)

    pop_d2 = analyze.population(d2[transient:], bins_d, sigma_d, weights[transient:], use_weights)
    fes_d2 = analyze.fes(pop_d2, kT)
    _, _, pop_list_d2 = analyze.bootstrap_pop(d2[transient:], bins_d, sigma_d, nb_bootstraps_1D, weights[transient:], use_weights)
    _, err_fes_d2, _ = analyze.error_fes(pop_list_d2, kT)

    pop_2D = analyze.population_2d(d1[transient:], d2[transient:], (bins_d, bins_d), (sigma_d, sigma_d), weights[transient:], use_weights)
    fes_2D = analyze.fes(pop_2D, kT)
    _, _, pop_list_2D = analyze.bootstrap_pop_2d(d1[transient:], d2[transient:], (bins_d, bins_d), (sigma_d, sigma_d), nb_bootstraps_2D, weights[transient:], use_weights)
    _, err_fes_2D, _ = analyze.error_fes(pop_list_2D, kT)

### Figures ###

if make_traj:
    if use_energy:
        figures.trj_E(Emec, av_Emec, std_Emec)
        figures.trj_T(Temp, av_Temp, std_Temp)

    figures.trj_dd(time, dd)
    figures.trj_d1(time, d1)
    figures.trj_d2(time, d2)
    
    figures.trj_2D(time, d1, d2)

    if use_opes:
        figures.trj_rct(time, rct)
        figures.trj_zed(time, zed)
        figures.trj_n(time, neff, nker)

    if use_chemcv:
        figures.trj_chemcv_charges(chemcv["q_Mulliken"].to_dict(), fixmin=True, fixmax=True, legend=False)
        figures.trj_chemcv_populations(chemcv["p_AtMO_Mulliken"].to_dict(), fixmin=True, fixmax=True, threshold=10, legend=False)
        figures.trj_chemcv_energies(chemcv["E_MO"].to_dict(), fixmin=True, fixmax=True, threshold=0.1, legend=False)
        figures.trj_chemcv(chemcv["sigma_MBIS"].to_dict(), ylabel="sigma_MBIS", fixmin=True, legend=False)

if make_density:
    figures.density_dd(grid_dd, density_dd)
    figures.density_d1(grid_d, density_d1)
    figures.density_d2(grid_d, density_d2)

    figures.density_2D(grid_d, density_2D)

if make_fes:
    figures.fes_dd(grid_dd, fes_dd, err_fes_dd)
    figures.fes_d1(grid_d, fes_d1, err_fes_d1)
    figures.fes_d2(grid_d, fes_d2, err_fes_d2)

    figures.fes_2D(grid_d, fes_2D)
    figures.err_fes_2D(grid_d, err_fes_2D)

plt.show()

if use_traj:
    if use_chemcv:
        figures.chemiscope_chemcv(traj, time, d1, d2, chemcv.to_dict(), chemcv["q_RESP"].to_numpy().T)
    else:
        figures.chemiscope(traj, time[traj_start:traj_end:traj_stride], d1[traj_start:traj_end:traj_stride], d2[traj_start:traj_end:traj_stride])