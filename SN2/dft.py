import os
os.chdir("SN2/TRAJ")

from ase.io import read
from ase.calculators.orca import ORCA

import numpy as np
import json
from rich.progress import Progress
import subprocess

traj = read("traj.xyz", ':')

nb_traj = len(traj)
nb_atoms = len(traj[0])

chemcv = np.zeros((nb_traj, 4*nb_atoms), dtype=np.float32)

progress = Progress()
task = progress.add_task("Processing...", total=nb_traj)
progress.start()
for i, atoms in enumerate(traj):
    subprocess.run("rm -rf ORCA", shell=True)
    atoms.calc = ORCA(charge=-1, mult=1, directory="ORCA", orcasimpleinput='WB97X-D4 def2-TZVPD', orcablocks="%pal nprocs 32 end") # PRINTMOS PRINTBASIS to visualize orbitals with .out
    e = atoms.get_potential_energy()
    subprocess.run("orca_2json ORCA/orca.gbw", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("orca_2json ORCA/orca -property", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm -f ORCA/orca.bibtex ORCA/orca.err ORCA/orca.inp ORCA/orca.densities ORCA/orca.densitiesinfo ORCA/orca.gbw ORCA/orca.JSON.bibtex ORCA/orca.property.txt", shell=True)
    with open("ORCA/orca.property.json", 'r') as f:
        properties = json.load(f)
    chemcv[i] = np.concatenate((np.array(properties["Geometries"][0]["Mulliken_Population_Analysis"][0]["AtomicCharges"]).T[0],
                                np.array(properties["Geometries"][0]["Loewdin_Population_Analysis"][0]["AtomicCharges"]).T[0],
                                np.array(properties["Geometries"][0]["Mayer_Population_Analysis"][0]["QA"]).T[0],
                                np.array(properties["Geometries"][0]["Mayer_Population_Analysis"][0]["VA"]).T[0]))
    progress.update(task, advance=1)
progress.stop()

header = ' '.join([' '.join([f"q_Mulliken.{i}" for i in range(nb_atoms)]),
                   ' '.join([f"q_Loewdin.{i}" for i in range(nb_atoms)]),
                   ' '.join([f"q_Mayer.{i}" for i in range(nb_atoms)]),
                   ' '.join([f"v_Mayer.{i}" for i in range(nb_atoms)])])
np.savetxt("CHEMCV", chemcv, delimiter=' ', fmt='%9.6f', header=header)
