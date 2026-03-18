import plumed

from ase import Atoms
from ase.build import molecule
from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read
from ase import units

from mace.calculators import mace_mp

import numpy as np

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
total_time = 50*units.fs
nb_steps = total_time//timestep

# Setup system
atoms = read("init.xyz")
atoms.set_cell([L, L, L])
atoms.set_pbc(True)
atoms.center()

# Setup MACE calculator
calc = mace_mp(model="small", device="cpu")

# Setup PLUMED OPES
input = open("plumed-metadynamics.dat", "r").read().splitlines() # STATE_WFILE=STATE STATE_WSTRIDE=10*100 STORE_STATES
plumed_calc = Plumed(calc, input, timestep, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep, T, taut)

# Extract useful quantities
interval=50
Epot, Ekin = np.empty(nb_steps//interval, dtype=np.float64), np.empty(nb_steps//interval, dtype=np.float64)
i = 0
def print_status(a=atoms):
    Epot[i], Ekin[i] = a.get_potential_energy()[0], a.get_kinetic_energy()
    i += 1
dyn.attach(print_status, interval)

# Run simulation
dyn.run(nb_steps)
