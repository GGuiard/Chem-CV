from ase.io import read
from mace.calculators import MACECalculator
import numpy as np
import os
from rich.progress import Progress

os.chdir("N2-Fe")

calc = MACECalculator('mace-Fe111-charges.model', model_type = 'EnergyChargesMACE')

traj = read("traj_comp.traj", ":")

nb_traj = len(traj)
nb_atoms = len(traj[0])

ref_q = np.append(8*np.ones(72), 5*np.ones(2))
charges = np.zeros((nb_traj, nb_atoms), dtype=np.float32)

progress = Progress()
task = progress.add_task("Processing...", total=nb_traj)
progress.start()
for i, atoms in enumerate(traj):
    atoms.calc = calc
    q = np.array(atoms.get_charges()) - ref_q
    q -= np.sum(q)/len(q)
    charges[i] = q
    progress.update(task, advance=1)
progress.stop()
 
np.savetxt("CHARGES", charges, delimiter=' ', fmt='%9.6f')  