import numpy as np
from ase import units
import plumed
import os

os.chdir("Ag-6_cluster")

import analyze
import figures

T = 100
kT = units.kB*T

time, c, r, logweights = plumed.read_as_pandas("COLVAR").to_numpy().T
weights = analyze.logw_to_w(logweights, kT)

bins_c, bins_r = np.linspace(np.min(c), np.max(c), 25), np.linspace(np.min(r), np.max(r), 25)
grid_c, grid_r = analyze.bin_to_grid(bins_c), analyze.bin_to_grid(bins_r)
pop = analyze.population_2d(r, c, (bins_r, bins_c), weights, use_weight=False)
fes = analyze.fes(pop, kT)

figures.fes_2D(grid_c, grid_r, fes)
