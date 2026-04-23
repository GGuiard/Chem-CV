import os
os.chdir("SN2/TRAJ")

from ase import Atoms
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase.optimize import QuasiNewton, MDMin
from ase.mep import NEB
from ase.io import write

from mace.calculators import mace_off

import numpy as np
import subprocess

subprocess.run("rm -f traj.xyz", shell=True)

images = []
for d1, d2 in zip([1.8, 1.8, 1.8, 1.9, 2, 2.5, 3, 3.5, 4], [4, 3.5, 3, 2.5, 2, 1.9, 1.8, 1.8, 1.8]):
    atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                        ( 0.000, 0.000,    d1),
                                        ( 0.000, 1.076, 0.000),
                                        ( 0.935,-0.540, 0.000),
                                        (-0.935,-0.540, 0.000),
                                        ( 0.000, 0.000,   -d2)])

    atoms.set_constraint([FixAtoms([0,1,5]), FixedLine(1, [0,0,1]), FixedPlane(2, [1,0,0])])

    atoms.calc = mace_off(model="large")
    dyn = QuasiNewton(atoms)
    dyn.run(fmax=0.05)

    atoms.set_constraint([FixAtoms(0), FixedLine([1,5], [0,0,1]), FixedPlane(2, [1,0,0])])
    images.append(atoms.copy())

neb = NEB(images)

for atoms in images:
    atoms.calc = mace_off(model="large")

optimizer = MDMin(neb)
optimizer.run(fmax=0.05)

d1, d2 = [], []
for atoms in images:
    d1.append(atoms.get_distance(0,5))
    d2.append(atoms.get_distance(0,1))

write('traj.xyz', images)
np.savetxt("COLVAR", np.array([np.arange(len(images)), np.array(d1)-np.array(d2), d1, d2]).T, delimiter=' ', fmt='%9.6f', header="! FIELDS time dd d1 d2", comments="#")