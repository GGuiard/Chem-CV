import os
os.chdir("SN2/TRAJ")

from ase.io import read
from ase.calculators.orca import ORCA

from rich.progress import Progress
import subprocess
from orca_parser import ChemCV

traj = read("traj.xyz", ':')

nb_traj = len(traj)

chemcv = ChemCV(nb_traj=nb_traj, 
                selections_per_type={"MO": [21,22], "atom": [0,1,5]},
                kwargs_per_cv={"q_AO_Mulliken": {"fmt": ["atom", "l"]},
                               "p_MOAO_Mulliken": {"fmt": ["MO", "atom", "l"]},
                               "q_AO_Loewdin": {"fmt": ["atom", "l"]},
                               "p_MOAO_Loewdin": {"fmt": ["MO", "atom", "l"]}})
simpleinput, blocks = chemcv.get_orca_input()

progress = Progress()
task = progress.add_task("Processing...", total=nb_traj)
progress.start()
for i, atoms in enumerate(traj):
    # PRINTMOS PRINTBASIS to visualize orbitals with .out
    atoms.calc = ORCA(charge=-1, mult=1, directory="ORCA", 
                      orcasimpleinput=' '.join(["WB97X-D4 def2-TZVPD", simpleinput]), 
                      orcablocks='\n'.join(["%pal nprocs 32 end", blocks]))
    _ = atoms.get_potential_energy()
    chemcv.update()
    subprocess.run("rm -rf ORCA", shell=True)
    progress.update(task, advance=1)
progress.stop()

print(chemcv.summary())
chemcv.save_hdf5("CHEMCV")