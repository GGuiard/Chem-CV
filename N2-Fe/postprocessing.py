import os
os.chdir("N2-Fe/unbiased")

import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.io import read
import plumed

import analyze
import figures

# Simulation parameters
T = 700
kT = units.kB*T

# Postprocessing parameters
transient = 0
nb_bins_d, nb_bins_c, nb_bins_q = 50, 50, 50
sigma_d, sigma_c, sigma_q = 0.1, 1, 0.1
nb_bootstraps_d, nb_bootstraps_c, nb_bootstraps_2D, nb_bootstraps_q = 10, 10, 10, 10
traj_start, traj_end, traj_stride = 0, 500000, 50

# Postprocessing options
use_energy = False
use_weights = False
use_com = False
use_opes = False
use_charge = False
use_charges_ref = False
use_traj = False

make_traj = True
make_density = False
make_av = True
make_fes = True

### Import data ###

if use_energy: Emec, Temp = np.loadtxt("ENERGY").T

data = plumed.read_as_pandas("COLVAR")
time, d, c = data[["time", "d", "c"]].to_numpy().T
if use_com: x, y, z = data[["com.x", "com.y", "com.z"]].to_numpy().T
if use_weights: weights = analyze.logw_to_w(data["opes.bias"].to_numpy().T, kT)
if use_opes: rct, zed, neff, nker = data[["opes.rct", "opes.zed", "opes.neff", "opes.nker"]].to_numpy().T
if use_charge: q = data["q.node-0"].to_numpy().T

if use_charges_ref:
    q_list_ref = np.loadtxt("CHARGES")
    q_ref = (q_list_ref[:,72]+q_list_ref[:,73])/2

if use_traj: traj = read("traj_comp.traj", f"{traj_start}:{traj_end}:{traj_stride}")

### Postprocessing ###

if not use_weights: weights = time

if use_energy:
    av_Emec, std_Emec = np.average(Emec), np.std(Emec)
    av_Temp, std_Temp = np.average(Temp), np.std(Temp)

if make_density or make_fes:
    bins_d, bins_c = np.linspace(np.min(d), 2, nb_bins_d), np.linspace(np.min(c), np.max(c), nb_bins_c)
    grid_d, grid_c = analyze.bin_to_grid(bins_d), analyze.bin_to_grid(bins_c)
    if use_charge:
        bins_q = np.linspace(np.min(q), np.max(q), nb_bins_q)
        grid_q = analyze.bin_to_grid(bins_q)

if make_density:
    density_d = analyze.population(d, bins_d, sigma_d)
    density_c = analyze.population(c, bins_c, sigma_c)
    if use_charge: density_q = analyze.population(q, bins_q, sigma_q)

    density_2D = analyze.population_2d(d, c, (bins_d, bins_c), (sigma_d, sigma_c))

if make_av:
    av_d, delta_d = analyze.cum_average(d[transient:], weights[transient:], use_weights)
    av_c, delta_c = analyze.cum_average(c[transient:], weights[transient:], use_weights)
    if use_charge: av_q, delta_q = analyze.cum_average(q[transient:], weights[transient:], use_weights)

if make_fes:
    pop_d = analyze.population(d[transient:], bins_d, sigma_d, weights[transient:], use_weights)
    fes_d = analyze.fes(pop_d, kT)
    _, err_pop_d, pop_list_d = analyze.bootstrap_pop(d[transient:], bins_d, sigma_d, nb_bootstraps_d, weights[transient:], use_weights)
    _, err_fes_d, _ = analyze.error_fes(pop_list_d, kT)

    pop_c = analyze.population(c[transient:], bins_c, sigma_c, weights[transient:], use_weights)
    fes_c = analyze.fes(pop_c, kT)
    _, _, pop_list_c = analyze.bootstrap_pop(c[transient:], bins_c, sigma_c, nb_bootstraps_c, weights[transient:], use_weights)
    _, err_fes_c, _ = analyze.error_fes(pop_list_c, kT)

    if use_charge:
        pop_q = analyze.population(q[transient:], bins_q, sigma_q, weights[transient:], use_weights)
        fes_q = analyze.fes(pop_q, kT)
        _, _, pop_list_q = analyze.bootstrap_pop(q[transient:], bins_q, sigma_q, nb_bootstraps_q, weights[transient:], use_weights)
        _, err_fes_q, _ = analyze.error_fes(pop_list_q, kT)

    pop_2D = analyze.population_2d(d[transient:], c[transient:], (bins_d, bins_c), (sigma_d, sigma_c), weights[transient:], use_weights)
    fes_2D = analyze.fes(pop_2D, kT)
    _, _, pop_list_2D = analyze.bootstrap_pop_2d(d[transient:], c[transient:], (bins_d, bins_c), (sigma_d, sigma_c), nb_bootstraps_2D, weights[transient:], use_weights)
    _, err_fes_2D, _ = analyze.error_fes(pop_list_2D, kT)

### Figures ###

if make_traj:
    if use_energy:
        figures.trj_E(Emec, av_Emec, std_Emec)
        figures.trj_T(Temp, av_Temp, std_Temp)

    figures.trj_d(time, d)
    figures.trj_c(time, c)
    if use_charge:
        figures.trj_q(time, q)
    
    figures.trj_2D(time, d, c)

    if use_com:
        figures.trj_z(time, z)
        figures.trj_xy(time, x, y)

    if use_opes:
        figures.trj_rct(time, rct)
        figures.trj_zed(time, zed)
        figures.trj_n(time, neff, nker)

if make_density:
    figures.density_d(grid_d, density_d)
    figures.density_c(grid_c, density_c)
    if use_charge: figures.density_q(grid_q, density_q)

    figures.density_2D(grid_d, grid_c, density_2D)

if make_av:
    figures.av_d(time[transient:], av_d)
    figures.av_c(time[transient:], av_c)
    figures.delta_d(time[transient:], delta_d)
    figures.delta_c(time[transient:], delta_c)

    if use_charge:
        figures.av_q(time[transient:], av_q)
        figures.delta_q(time[transient:], delta_q)

if make_fes:
    figures.fes_d(grid_d, fes_d, err_fes_d)
    figures.fes_c(grid_c, fes_c, err_fes_c)
    if use_charge: figures.fes_q(grid_q, fes_q, err_fes_q)

    figures.fes_2D(grid_d, grid_c, fes_2D)
    figures.err_fes_2D(grid_d, grid_c, err_fes_2D)

if use_charges_ref and use_charge: figures.pred(q_ref, q)

plt.show()

if use_traj:
    if not use_charge: figures.chemiscope(traj, time[traj_start:traj_end:traj_stride], d[traj_start:traj_end:traj_stride], c[traj_start:traj_end:traj_stride])
    if use_charge: figures.chemiscope_charge(traj, d[traj_start:traj_end:traj_stride], c[traj_start:traj_end:traj_stride], q[traj_start:traj_end:traj_stride])
    if use_charges_ref: figures.chemiscope_charges(traj, d[traj_start:traj_end:traj_stride], c[traj_start:traj_end:traj_stride], q_list_ref[traj_start:traj_end:traj_stride])
