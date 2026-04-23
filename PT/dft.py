import os
os.chdir("PT/TRAJ")

from ase.io import read
from ase.calculators.orca import ORCA

from rich.progress import Progress
import subprocess
from orca_parser import ChemCV

traj = read("traj.xyz", ':')

nb_traj = len(traj)
nb_atoms = len(traj[0])

chemcv = ChemCV(nb_traj, nb_atoms)
simpleinput, blocks = chemcv.get_orca_input()

progress = Progress()
task = progress.add_task("Processing...", total=nb_traj)
progress.start()
for i, atoms in enumerate(traj):
    atoms.calc = ORCA(charge=0, mult=1, directory="ORCA", orcasimpleinput="WB97X-D4 def2-TZVP"+simpleinput, orcablocks="%pal nprocs 32 end"+blocks) # PRINTMOS PRINTBASIS to visualize orbitals with .out
    _ = atoms.get_potential_energy()
    chemcv.update()
    subprocess.run("rm -rf ORCA", shell=True)
    progress.update(task, advance=1)
progress.stop()

chemcv.save()