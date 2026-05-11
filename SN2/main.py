import os
os.chdir("SN2")

from ase.calculators.plumed import Plumed, restart_from_trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read, Trajectory
from ase import units

from mace.calculators import mace_off

import subprocess
from tqdm import tqdm

# Simulation parameters
T = 300 # K
kT = units.kB*T
timestep = 0.5 # fs
taut = 100 # fs
total_time = 1000000 # fs
nb_steps = int(total_time//timestep)
interval_info = int(nb_steps//100)
interval_traj = 10 # must be a multiple of the plumed stride
restart, prev_steps = False, 1000000

# Clean
if not restart:
    subprocess.run("rm -f bck.* *.traj COLVAR KERNELS STATES HILLS ENERGY", shell=True)

# Setup system
atoms = read("init.xyz")

# Setup MACE calculator
calc = mace_off(model="https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off24/MACE-OFF24_medium.model", default_dtype="float32")

# Setup PLUMED
input = open("plumed-opes.dat", "r").read().splitlines()
if restart:
    plumed_calc = restart_from_trajectory(prev_traj="traj_comp.traj",
                                          prev_steps=prev_steps,
                                          calc=calc,
                                          input=input,
                                          timestep=timestep*units.fs,
                                          atoms=atoms,
                                          kT=kT)
else:
    plumed_calc = Plumed(calc, input, timestep*units.fs, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep*units.fs, T, taut*units.fs)

# Recenter
dyn.attach(lambda: atoms.center())

# Extract useful quantities
if not restart:
    with open("ENERGY", 'w') as f:
        f.write("#! FIELDS Emec Temp\n")

def print_status():
    Emec = atoms.calc.results['energy'][0] + atoms.get_kinetic_energy()
    Temp = atoms.get_temperature()

    with open("ENERGY", 'a') as f:
        f.write(f"{Emec:9.6f} {Temp:9.6f}\n")

dyn.attach(print_status, interval_info)

# Save trajectory
traj = Trajectory("traj_comp.traj", 'w', atoms)
dyn.attach(traj, interval_traj)

# Setup progress bar
pbar = tqdm(total=101, unit="step")

def update_progress():
    pbar.update()

dyn.attach(update_progress, int(nb_steps//100))

# Run simulation
dyn.run(nb_steps)

pbar.close()