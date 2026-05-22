import os
import subprocess

directory = "SN2"
os.chdir(directory)

from ase.io import read, write
from ase.calculators.orca import ORCA
from tqdm import tqdm
import contextlib

from orca_parser import ChemCV

traj = read("SAMPLING/sampling.xyz", ':', format="xyz")

for atoms in traj:
    atoms.positions -= atoms.get_center_of_mass()

write("DFT/sampling.xyz", traj, format="xyz")

nb_traj = len(traj)

chemcv = ChemCV(
    selections_per_type={"MO": [21,22], "atom": [0,1,5], "l": "p"},
    kwargs_per_cv={"q_AO_Mulliken": {"fmt": ["atom", "l"]},
                   "p_MOAO_Mulliken": {"fmt": ["MO", "atom", "l"]},
                   "q_AO_Loewdin": {"fmt": ["atom", "l"]},
                   "p_MOAO_Loewdin": {"fmt": ["MO", "atom", "l"]}},
    nb_traj=nb_traj,
)

simpleinput, blocks = chemcv.get_orca_input()

calc = ORCA(
    charge=-1,
    mult=1,
    directory="ORCA", 
    orcasimpleinput=' '.join(["WB97X-D4 def2-TZVPD EnGrad", simpleinput]), 
    orcablocks='\n'.join(["%pal nprocs 32 end", blocks]),
)

for atoms in tqdm(traj, desc=directory):
    atoms.calc = calc

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                _ = atoms.get_potential_energy()
                write("SN2/traj.xyz", atoms, append=True)
                chemcv.update()

    subprocess.run("rm -rf ORCA", shell=True)

chemcv.save(format="json")