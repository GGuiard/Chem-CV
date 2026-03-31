from ase.calculators.idealgas import IdealGas
from ase.calculators.plumed import Plumed
from ase.io import read
from ase import units
import os

os.chdir("N2-Fe")

# Simulation parameters
T = 700 # K
kT = units.kB*T
timestep = 0.5 # fs

# Setup system
atoms = read("init.xyz")

# Import trajectory
traj = read("traj_comp.traj", ":")

# Setup PLUMED
input = open("plumed-descriptors.dat", "r").read().splitlines()
plumed_calc = Plumed(IdealGas, input, timestep*units.fs, atoms, kT)

# Extract descriptors
plumed_calc.write_plumed_files(traj)
