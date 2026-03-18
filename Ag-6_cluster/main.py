import plumed

from ase import Atoms
from ase.build import molecule
from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read
from ase import units

from mace.calculators import mace_mp

import os
import subprocess

os.chdir("Ag-6_cluster")
subprocess.run("rm -f bck.* COLVAR KERNELS STATE", shell=True)

# Simulation parameters
T = 100
L = 16
kT = units.kB*T
timestep = 5*units.fs
taut = 50*units.fs

# Setup system
atoms = read("init.xyz")
atoms.set_cell([L, L, L])
atoms.set_pbc(True)
atoms.center()

# Setup MACE calculator
calc = mace_mp(model="small", device="cpu")

# Setup PLUMED OPES
input = open("plumed.dat", "r").read().splitlines()
plumed_calc = Plumed(calc, input, timestep, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep, T, taut)

# Run simulation
dyn.run(10000)
