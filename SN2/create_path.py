import os
os.chdir("SN2/TRAJ")

from ase import Atoms
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase.optimize import QuasiNewton
from ase.md.verlet import VelocityVerlet
from ase.io import read, write

from mace.calculators import mace_off

import numpy as np
import subprocess
import figures

subprocess.run("rm -f traj.xyz", shell=True)

### By hand method ###
images = []
d1, d2 = [], []

for d in np.linspace(3.95, 2.15, 19):
    atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                        ( 0.000, 0.000,     d),
                                        ( 0.000, 1.075, 0.000),
                                        ( 0.931,-0.538, 0.000),
                                        (-0.931,-0.538, 0.000),
                                        ( 0.000, 0.000,-2.040)])
    
    atoms.calc = mace_off(model="large")
    atoms.set_constraint([FixAtoms([0,1]), FixedLine(5, [0,0,1])])

    dyn = QuasiNewton(atoms)
    dyn.run()

    images.append(atoms.copy())
    d1.append(atoms.get_distance(0,5))
    d2.append(atoms.get_distance(0,1))

atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                    ( 0.000, 0.000, 2.040),
                                    ( 0.000, 1.075, 0.000),
                                    ( 0.931,-0.538, 0.000),
                                    (-0.931,-0.538, 0.000),
                                    ( 0.000, 0.000,-2.040)])
    
atoms.calc = mace_off(model="large")

images.append(atoms.copy())
d1.append(atoms.get_distance(0,5))
d2.append(atoms.get_distance(0,1))
    
for d in np.linspace(2.15, 3.95, 19):
    atoms = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
                                        ( 0.000, 0.000, 2.040),
                                        ( 0.000, 1.075, 0.000),
                                        ( 0.931,-0.538, 0.000),
                                        (-0.931,-0.538, 0.000),
                                        ( 0.000, 0.000,    -d)])
    
    atoms.calc = mace_off(model="large")
    atoms.set_constraint([FixAtoms([0,5]), FixedLine(1, [0,0,1])])

    dyn = QuasiNewton(atoms)
    dyn.run()

    images.append(atoms.copy())
    d1.append(atoms.get_distance(0,5))
    d2.append(atoms.get_distance(0,1))

### MD method ###

# ts = Atoms("CClH3Cl", positions=[( 0.000, 0.000, 0.000),
#                                  ( 0.000, 0.000, 2.093),
#                                  ( 0.000, 1.075, 0.000),
#                                  ( 0.931,-0.538, 0.000),
#                                  (-0.931,-0.538, 0.000),
#                                  ( 0.000, 0.000,-2.093)])

# v0 = 0.2

# atoms = ts.copy()
# atoms.set_velocities([[0., 0., 0.],
#                       [0., 0., v0],
#                       [0., 0., 0.],
#                       [0., 0., 0.],
#                       [0., 0., 0.],
#                       [0., 0., v0]])

# atoms.calc = mace_off(model="large")
# atoms.set_constraint([FixAtoms(0), FixedLine([1,5], [0,0,1]), FixedPlane(2, [1,0,0])])
# dyn = VelocityVerlet(atoms, timestep=0.3, trajectory="traj.xyz")
# dyn.run(steps=10)

# trj1 = read("traj.xyz", ":")

# atoms = ts.copy()
# atoms.set_velocities([[0., 0., 0.],
#                       [0., 0.,-v0],
#                       [0., 0., 0.],
#                       [0., 0., 0.],
#                       [0., 0., 0.],
#                       [0., 0.,-v0]])

# atoms.calc = mace_off(model="large")
# atoms.set_constraint([FixAtoms(0), FixedLine([1,5], [0,0,1]), FixedPlane(2, [1,0,0])])
# dyn = VelocityVerlet(atoms, timestep=0.3, trajectory="traj.xyz")
# dyn.run(steps=10)

# trj2 = read("traj.xyz", ":")

# images = trj1[:0:-1] + [ts] + trj2[1:]

# for atoms in images:
#     atoms.calc = mace_off(model="large")

# d1, d2 = [], []
# for atoms in images:
#     d1.append(atoms.get_distance(0,5))
#     d2.append(atoms.get_distance(0,1))


time = np.arange(len(images))
dd = np.array(d1)-np.array(d2)

figures.chemiscope(images, time, d1, d2)

write('path.xyz', images)
np.savetxt("COLVAR", np.array([time, dd, d1, d2]).T, delimiter=' ', fmt='%9.6f', header="FIELDS time dd d1 d2", comments="#! ")