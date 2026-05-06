import os
os.chdir("SN2")

from ase.calculators.plumed import Plumed, restart_from_trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase.io import read, Trajectory
from ase import units

from mace.calculators import mace_off

import subprocess
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

# Simulation parameters
T = 300 # K
kT = units.kB*T
timestep = 0.5 # fs
taut = 100 # fs
total_time = 1000 # fs
nb_steps = int(total_time//timestep)
interval_info = 1 # int(nb_steps//100)
interval_traj = 1 # must be a multiple of the plumed stride
restart, prev_steps = False, 1000000

# Clean
if not restart:
    subprocess.run("rm -f bck.* *.traj COLVAR KERNELS STATES HILLS ENERGY", shell=True)

# Setup system
atoms = read("init.xyz")

# Setup MACE calculator
calc = mace_off(model="large")

# Setup PLUMED
input = open("plumed-opes.dat", "r").read().splitlines()
if restart:
    plumed_calc = restart_from_trajectory(prev_traj="traj_comp.traj", prev_steps=prev_steps, calc=calc, input=input, timestep=timestep*units.fs, atoms=atoms, kT=kT)
else:
    plumed_calc = Plumed(calc, input, timestep*units.fs, atoms, kT)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
dyn = Bussi(atoms, timestep*units.fs, T, taut*units.fs)

# Extract useful quantities
if not restart:
    with open("ENERGY", 'w') as f:
        f.write("#! FIELDS Emec Temp\n")

def print_status(a=atoms):
    Emec = a.calc.results['energy'][0] + a.get_kinetic_energy()
    Temp = a.get_temperature()

    with open("ENERGY", 'a') as f:
        f.write(f"{Emec:9.6f} {Temp:9.6f}\n")

dyn.attach(print_status, interval_info)

# Save trajectory
traj = Trajectory("traj_comp.traj", 'w', atoms)
dyn.attach(traj, interval_traj)

# Setup progress bar
progress = Progress(BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn())
task = progress.add_task("Processing...", total=100)

def update_progress():
    progress.update(task, advance=1)

dyn.attach(update_progress, int(nb_steps//100))

# Diagnose
# import time

# with open("TIME", 'w') as f:
#     f.write("#! FIELDS time\n")
# last = [time.time()]

# def diagnose(a=atoms):
#     new = time.time()
#     with open("TIME", 'a') as f:
#         f.write(f"{new-last[0]:9.6f}\n")
#     last[0] = new

# dyn.attach(diagnose, int(nb_steps//100))

# Run simulation
progress.start()
dyn.run(nb_steps)
progress.stop()