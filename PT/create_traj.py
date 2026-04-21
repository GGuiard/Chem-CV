import os
os.chdir("PT")

from ase import Atoms
from ase.constraints import FixAtoms, FixedPlane, FixBondLength
from ase.optimize import QuasiNewton
from ase.io import write

from mace.calculators import mace_off

import numpy as np
import subprocess

subprocess.run("rm -f traj.xyz", shell=True)

calc = mace_off(model="large")

d1, d2 = [], []
for d in np.linspace(0, 1, 10):

    atoms = Atoms("C3O2H4", positions=[(-1.178, 0.811, 0.000),
                                       ( 0.000, 0.059, 0.000),
                                       ( 1.178, 0.811, 0.000),
                                       (-1.168, 2.064, 0.000),
                                       ( 1.168, 2.064, 0.000),
                                       (     d, 2.321+d, 0.000),
                                       (-2.143, 0.321, 0.000),
                                       ( 0.000,-1.011, 0.000),
                                       ( 2.143, 0.321, 0.000)])

    atoms.set_constraint([FixAtoms(1), FixedPlane([0,1,2,3,4,5,6,7,8], [0,0,1]), FixBondLength(3, 5)])

    atoms.calc = calc
    dyn = QuasiNewton(atoms)
    dyn.run(fmax=0.05)

    write('traj.xyz', atoms, append='a')
    
    d1.append(atoms.get_distance(3,5))
    d2.append(atoms.get_distance(4,5))

np.savetxt("COLVAR", np.array([np.arange(10), np.array(d1)-np.array(d2), d1, d2]).T, delimiter=' ', fmt='%9.6f', header="! FIELDS time dd d1 d2")