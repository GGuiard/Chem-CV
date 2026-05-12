import os
os.chdir("SN2")

from ase import Atoms
from ase.build import molecule
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase.io import write

from mace.calculators import mace_off

import subprocess

subprocess.run("rm -f input.xyz", shell=True)

atoms = molecule("CH3Cl")
atoms += Atoms("Cl", positions=[(0,0,-3)])
atoms.center(vacuum=100)
atoms.set_pbc((True, True, True))

calc = mace_off(model="https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off24/MACE-OFF24_medium.model", default_dtype="float64")
atoms.calc = calc
atoms.set_constraint([FixAtoms(0), FixedLine([1,5], [0,0,1]), FixedPlane(2, [1,0,0])])

dyn = QuasiNewton(atoms)
dyn.run(fmax=0.05)

write('init.xyz', atoms)