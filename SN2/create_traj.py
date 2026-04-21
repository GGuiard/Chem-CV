import os
os.chdir("SN2")

from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase.optimize import QuasiNewton
from ase.io import write

from mace.calculators import mace_off

import numpy as np
import subprocess

subprocess.run("rm -f traj.xyz", shell=True)

calc = mace_off(model="large")

d1, d2 = [], []
for d in np.linspace(-4.3, -3.3, 10):

    atoms = molecule("CH3Cl")
    atoms += Atoms("Cl", positions=[(0,0,d)])

    atoms.set_constraint([FixAtoms([0,5]), FixedLine(1, [0,0,1]), FixedPlane(2, [1,0,0])])

    atoms.calc = calc
    dyn = QuasiNewton(atoms)
    dyn.run(fmax=0.05)

    write('traj.xyz', atoms, append='a')
    
    d1.append(atoms.get_distance(0,5))
    d2.append(atoms.get_distance(0,1))

np.savetxt("COLVAR", np.array([np.arange(10), np.array(d1)-np.array(d2), d1, d2]).T, delimiter=' ', fmt='%9.6f', header="! FIELDS time dd d1 d2")