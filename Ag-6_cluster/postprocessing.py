import numpy as np
import matplotlib.pyplot as plt
from ase import units
import plumed
import os

os.chdir("Ag-6_cluster")

import analyze
import figures

# Simulation parameters
T = 100
kT = units.kB*T

### Import data ###

time, c, r, logweights = plumed.read_as_pandas("COLVAR").to_numpy().T
weights, use_weights = analyze.logw_to_w(logweights, kT), True

# time, c, r = plumed.read_as_pandas("COLVAR").to_numpy().T
# weights, use_weights = None, False

### Postprocessing ###

av_c, delta_c = analyze.cum_average(c, weights, use_weights)
av_r, delta_r = analyze.cum_average(r, weights, use_weights)

bins_c, bins_r = np.linspace(np.min(c), np.max(c), 10), np.linspace(np.min(r), np.max(r), 10)
grid_c, grid_r = analyze.bin_to_grid(bins_c), analyze.bin_to_grid(bins_r)

pop_c = analyze.population(c, bins_c, weights, use_weights)
fes_c = analyze.fes(pop_c, kT)
_, _, pop_list_c = analyze.bootstrap_pop(c, bins_c, 10, weights, use_weights)
_, err_fes_c, _ = analyze.error_fes(pop_list_c, kT)

pop_r = analyze.population(r, bins_r, weights, use_weights)
fes_r = analyze.fes(pop_r, kT)
_, _, pop_list_r = analyze.bootstrap_pop(r, bins_r, 10, weights, use_weights)
_, err_fes_r, _ = analyze.error_fes(pop_list_r, kT)

pop_2D = analyze.population_2d(c, r, (bins_c, bins_r), weights, use_weights)
fes_2D = analyze.fes(pop_2D, kT)
_, _, pop_list_2D = analyze.bootstrap_pop_2d(c, r, (bins_c, bins_r), 10, weights, use_weights)
_, err_fes_2D, _ = analyze.error_fes(pop_list_2D, kT)

### Figures ###

figures.av_c(time, av_c)
figures.av_r(time, av_r)
figures.delta_c(time, delta_c)
figures.delta_r(time, delta_r)

figures.trj_c(time, c)
figures.trj_r(time, r)
figures.trj_2D(c, r)

figures.fes_c(grid_c, fes_c, err_fes_c)
figures.fes_r(grid_r, fes_r, err_fes_r)
figures.fes_2D(grid_c, grid_r, fes_2D)
figures.err_fes_2D(grid_c, grid_r, err_fes_2D)

plt.show()
