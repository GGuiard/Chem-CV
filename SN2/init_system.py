import os
os.chdir("SN2")

from ase import Atoms
from ase.build import molecule
from ase.optimize import QuasiNewton
from ase.io import write

from mace.calculators import mace_off

import subprocess

subprocess.run("rm -f input.xyz", shell=True)

calc = mace_off(model="large")

atoms = molecule("CH3Cl")
atoms += Atoms("Cl", positions=[(0,0,-3)])

atoms.calc = calc
dyn = QuasiNewton(atoms)
dyn.run(fmax=0.05)

atoms.center(vacuum=6.0)

write('init.xyz', atoms)