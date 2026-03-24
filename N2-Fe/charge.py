from ase.io import read

from mace.calculators import MACECalculator

calc = MACECalculator('mace-Fe111-charges.model', model_type = 'EnergyChargesMACE')

traj = read("traj_comp.traj", ":")

for atoms in traj:
    q = atoms.get_charges()
    print(q)
 