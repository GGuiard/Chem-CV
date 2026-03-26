import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.io import read
import plumed
import os

os.chdir("N2-Fe")

import analyze
import figures

# Simulation parameters
T = 700
kT = units.kB*T

### Import data ###

time, d, c, x, y, z, logweights, rct, zed, neff, nker = plumed.read_as_pandas("COLVAR").to_numpy()[::10].T
weights, use_weights = analyze.logw_to_w(logweights, kT), True

# time, d, c, x, y, z, logweights = plumed.read_as_pandas("COLVAR").to_numpy().T
# weights, use_weights = analyze.logw_to_w(logweights, kT), True

# time, d, c, x, y, z = plumed.read_as_pandas("COLVAR").to_numpy().T
# weights, use_weights = None, False

traj = read("traj_comp.traj", "::10")

q = np.loadtxt("CHARGES")

### Postprocessing ###

av_d, delta_d = analyze.cum_average(d, weights, use_weights)
av_c, delta_c = analyze.cum_average(c, weights, use_weights)

bins_d, bins_c = np.linspace(np.min(d), 2, 100), np.linspace(np.min(c), np.max(c), 100)
grid_d, grid_c = analyze.bin_to_grid(bins_d), analyze.bin_to_grid(bins_c)

pop_d = analyze.population(d, bins_d, weights, use_weights)
fes_d = analyze.fes(pop_d, kT)
_, _, pop_list_d = analyze.bootstrap_pop(d, bins_d, 10, weights, use_weights)
_, err_fes_d, _ = analyze.error_fes(pop_list_d, kT)

pop_c = analyze.population(c, bins_c, weights, use_weights)
fes_c = analyze.fes(pop_c, kT)
_, _, pop_list_c = analyze.bootstrap_pop(c, bins_c, 10, weights, use_weights)
_, err_fes_c, _ = analyze.error_fes(pop_list_c, kT)

pop_2D = analyze.population_2d(d, c, (bins_d, bins_c), weights, use_weights)
fes_2D = analyze.fes(pop_2D, kT)
_, _, pop_list_2D = analyze.bootstrap_pop_2d(d, c, (bins_d, bins_c), 10, weights, use_weights)
_, err_fes_2D, _ = analyze.error_fes(pop_list_2D, kT)

## Figures ###

figures.av_d(time, av_d)
figures.av_c(time, av_c)
figures.delta_d(time, delta_d)
figures.delta_c(time, delta_c)

figures.trj_d(time, d)
figures.trj_c(time, c)
figures.trj_2D(d, c)

figures.trj_z(time, z)
figures.trj_xy(x, y)

figures.fes_d(grid_d, fes_d, err_fes_d)
figures.fes_c(grid_c, fes_c, err_fes_c)
figures.fes_2D(grid_d, grid_c, fes_2D)
figures.err_fes_2D(grid_d, grid_c, err_fes_2D)

plt.show()

figures.chemiscope(traj, time, d, c)

figures.chemiscope_charges(traj, d, c, q)
