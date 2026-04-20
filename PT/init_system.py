import os
os.chdir("PT")

from ase import Atoms
from ase.optimize import QuasiNewton
from ase.io import write

from mace.calculators import mace_off

import subprocess

subprocess.run("rm -f input.xyz", shell=True)

calc = mace_off(model="large")

atoms = Atoms("C3O2H4", positions=[(-1.178, 0.811, 0.000),
                                   ( 0.000, 0.059, 0.000),
                                   ( 1.178, 0.811, 0.000),
                                   (-1.168, 2.064, 0.000),
                                   ( 1.168, 2.064, 0.000),
                                   ( 0.500, 2.321, 0.000),
                                   (-2.143, 0.321, 0.000),
                                   ( 0.000,-1.011, 0.000),
                                   ( 2.143, 0.321, 0.000)])

atoms.calc = calc
dyn = QuasiNewton(atoms)
dyn.run(fmax=0.05)

atoms.center(vacuum=6.0)

write('init.xyz', atoms)