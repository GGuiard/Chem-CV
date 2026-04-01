import os
os.chdir("ExtraCV")

from ase import Atoms
from ase.io import write

atoms = Atoms('2H', positions=[(0,0,0), (1,0,0)])

write('init.xyz', atoms)
