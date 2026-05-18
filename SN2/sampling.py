import os
os.chdir("SN2/OPES")

import numpy as np
import plumed
from ase.io import read, write
import utils.figures
import utils.helpers as helpers

def bin_sampling(
    cv: np.ndarray,
    nb_bins: int = 20,
    nb_per_bin: int = 50,
    bounds: tuple = (None, None),
) -> np.ndarray:
    if not bounds[0]:
        bounds[0] = np.min(cv)
    if not bounds[1]:
        bounds[1] = np.max(cv)

    bins = np.linspace(bounds[0], bounds[1], nb_bins+1)

    indices_per_bin = []
    for bin in zip(bins[:-1], bins[1:]):
        arg = np.argwhere(np.logical_and(bin[0]<cv, cv<bin[1])).T[0]
        sample = np.random.choice(arg, nb_per_bin, replace=False)
        indices_per_bin.append(sample)

    return np.array(indices_per_bin).ravel()

data = plumed.read_as_pandas("COLVAR").iloc[::10]
dd = data["dd"].to_numpy().T

indices = bin_sampling(dd, bounds=(-2.5, 2.5))
np.save("sampling_indices", indices)

structures = []
for index in indices:
    atoms = read("traj_comp.traj", index)
    atoms.positions -= atoms.get_center_of_mass()
    structures.append(atoms)
write("sampling.xyz", structures, format="xyz")

plot_structures = helpers.orient_traj(structures, y_axis=5)
dd = dd[indices]
d1 = data["d1"].to_numpy().T[indices]
d2 = data["d2"].to_numpy().T[indices]
utils.figures.chemiscope(plot_structures, dd, d1, d2)