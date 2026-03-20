from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read
from ase import units

from mace.calculators import mace_mp

import numpy as np
import matplotlib.pyplot as plt

import os
import subprocess

os.chdir("N2-Fe")
subprocess.run("rm -f bck.* COLVAR KERNELS STATE HILLS", shell=True)

import analyze
import figures

# Simulation parameters
T = 700 # K
kT = units.kB*T
timestep = 0.5 # fs
taut = 100 # fs
total_time = 1000 # fs
nb_steps = int(total_time//timestep)

# Setup system
atoms = read("init.xyz")
nb_atoms = len(atoms)

# Setup MACE calculator
calc = mace_mp(model='mh-0', head='oc20_usemppbe')

# Setup PLUMED OPES
input = open("plumed-unbiased.dat", "r").read().splitlines() # NLIST NL_CUTOFF=5, NL_STRIDE=100
plumed_calc = Plumed(calc, input, timestep*units.fs, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep*units.fs, T, taut*units.fs)

# Extract useful quantities
interval=50
Epot, Ekin = np.empty(int(nb_steps//interval)+1, dtype=np.float64), np.empty(int(nb_steps//interval)+1, dtype=np.float64)
i = 0
def print_status(a=atoms):
    global i # to change
    Epot[i], Ekin[i] = a.get_potential_energy()[0], a.get_kinetic_energy()
    i += 1
dyn.attach(print_status, interval)

# Run simulation
dyn.run(nb_steps)

# Analyze
Emec, av_Emec, std_Emec = analyze.Emec(Epot, Ekin)
Temp, av_Temp, std_Temp = analyze.T(Ekin, nb_atoms)

figures.trj_E(Epot, Emec, av_Emec, std_Emec)
figures.trj_T(Temp, av_Temp, std_Temp)
plt.show()