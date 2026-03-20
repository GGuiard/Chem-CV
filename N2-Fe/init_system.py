from ase import Atoms
from ase.build import add_adsorbate, bcc111
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.io import write
from ase.visualize import view

from mace.calculators import mace_mp

import numpy as np

import os
import subprocess

os.chdir("N2-Fe")
subprocess.run("rm -f input.xyz opt.traj", shell=True)

l_Ncenter, d_N2, h_NFe, z_Fe = 1, 1.24, 1.15, 3

slab = bcc111('Fe', size=(3,3,8))

calc = mace_mp(model='mh-0', head='oc20_usemppbe', default_dtype='float64')
slab.calc = calc
slab.set_pbc((1,1,0))
slab.set_constraint(FixAtoms(mask=[a.z<z_Fe for a in slab]))

dyn = QuasiNewton(slab)
dyn.run(fmax=0.05)

molecule = Atoms('2N', positions=[(0,0,0), (d_N2,0,0)])
add_adsorbate(slab, molecule, h_NFe, position=(-d_N2/2,0), offset=(9/6,1))

dyn = QuasiNewton(slab, trajectory='opt.traj')
dyn.run(fmax=0.05)

view(slab)

write('input.xyz', slab)
