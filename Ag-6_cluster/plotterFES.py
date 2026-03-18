import matplotlib.pyplot as plt
from matplotlib import colormaps
# import cmocean.cm as cmo
import numpy as np
from ase import units
import plumed
import os 
import subprocess

os.chdir("Ag-6_cluster")

import analyze

T = 100
kT = units.kB*T

time, c, r, logweight = plumed.read_as_pandas("COLVAR").to_numpy().T
weights = analyze.logw_to_w(logweight, kT)

bins_c, bins_r = np.linspace(np.min(c), np.max(c), 50), np.linspace(np.min(r), np.max(r), 50)
grid_c, grid_r = analyze.bin_to_grid(bins_c), analyze.bin_to_grid(bins_r)
pop = analyze.population_2d(r, c, (bins_r, bins_c), weights, use_weight=True)
fes = analyze.fes(pop, kT)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.contourf(grid_c, grid_r, fes, 10, cmap=colormaps['Blues_r'])# cmo.tempo_r)
cp = ax.contour(grid_c, grid_r, fes, 10, linestyles='-', colors='darkgray', linewidths=1.2)

ax.set_xlabel('Coordination', fontsize=40)
ax.set_ylabel('Gyration', fontsize=40)
ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='x', labelsize=25)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label='FES [eV]', fontsize=40)
cbar.ax.tick_params(labelsize=32)

plt.tight_layout()
plt.show()
