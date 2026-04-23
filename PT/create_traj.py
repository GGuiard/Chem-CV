import os
os.chdir("PT/TRAJ")

from ase import Atoms
from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import QuasiNewton
from ase.io import write, read

from mace.calculators import mace_off

import numpy as np
import subprocess

subprocess.run("rm -f traj.xyz", shell=True)

ts = atoms = Atoms("C3O2H4", positions=[(-1.178, 0.811, 0.000),
                                        ( 0.000, 0.059, 0.000),
                                        ( 1.178, 0.811, 0.000),
                                        (-1.168, 2.064, 0.000),
                                        ( 1.168, 2.064, 0.000),
                                        ( 0.000, 2.321, 0.000),
                                        (-2.143, 0.321, 0.000),
                                        ( 0.000,-1.011, 0.000),
                                        ( 2.143, 0.321, 0.000)])

atoms = Atoms("C3O2H4", positions=[(-1.178, 0.811, 0.000),
                                   ( 0.000, 0.059, 0.000),
                                   ( 1.178, 0.811, 0.000),
                                   (-1.168, 2.064, 0.000),
                                   ( 1.168, 2.064, 0.000),
                                   ( 0.010, 2.321, 0.000),
                                   (-2.143, 0.321, 0.000),
                                   ( 0.000,-1.011, 0.000),
                                   ( 2.143, 0.321, 0.000)])

atoms.calc = mace_off(model="large")
atoms.set_constraint([FixAtoms([1]), FixedPlane([0,1,2,3,4,5,6,7,8], [0,0,1])])
dyn = QuasiNewton(atoms, trajectory="traj.xyz")
dyn.run(steps=10)

irc1 = read("traj.xyz", ":")

atoms = Atoms("C3O2H4", positions=[(-1.178, 0.811, 0.000),
                                   ( 0.000, 0.059, 0.000),
                                   ( 1.178, 0.811, 0.000),
                                   (-1.168, 2.064, 0.000),
                                   ( 1.168, 2.064, 0.000),
                                   (-0.010, 2.321, 0.000),
                                   (-2.143, 0.321, 0.000),
                                   ( 0.000,-1.011, 0.000),
                                   ( 2.143, 0.321, 0.000)])

atoms.calc = mace_off(model="large")
atoms.set_constraint([FixAtoms([1]), FixedPlane([0,1,2,3,4,5,6,7,8], [0,0,1])])
dyn = QuasiNewton(atoms, trajectory="traj.xyz")
dyn.run(steps=10)

irc2 = read("traj.xyz", ":")

traj = irc1[::-1] + [ts] + irc2

time, d1, d2 = [], [], []
for i, atoms in enumerate(traj):
    time.append(i)
    d1.append(atoms.get_distance(3,5))
    d2.append(atoms.get_distance(4,5))

write('traj.xyz', traj)
np.savetxt("COLVAR", np.array([time, np.array(d1)-np.array(d2), d1, d2]).T, delimiter=' ', fmt='%9.6f', header="! FIELDS time dd d1 d2", comments="#")