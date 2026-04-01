import os
os.chdir("ExtraCV")

from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read, Trajectory
from ase import units
from ase.constraints import FixedLine

from mace.calculators import mace_mp

import subprocess

# Simulation parameters
T = 300 # K
kT = units.kB*T
timestep = 0.2 # fs
taut = 100 # fs
total_time = 10 # fs
nb_steps = int(total_time//timestep)

# Clean
subprocess.run("rm -f bck.* *.traj COLVAR", shell=True)

# Setup system
atoms = read("init.xyz")
atoms.set_constraint(FixedLine([0,1], [1,0,0]))

# Setup MACE calculator
calc = mace_mp(model='mh-0', head='oc20_usemppbe')

# Setup PLUMED
input = open("plumed.dat", "r").read().splitlines()
plumed_calc = Plumed(calc, input, timestep*units.fs, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep*units.fs, T, taut*units.fs)

# Save trajectory
traj = Trajectory("traj_comp.traj", 'w', atoms)
dyn.attach(traj)

# Run simulation
dyn.run(nb_steps)
