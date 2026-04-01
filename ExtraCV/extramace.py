from mace.calculators import MACECalculator
from ase.calculators.calculator import all_changes

class MyCalculator(MACECalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
         
    def calculate(self, extracv, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["cv"], self.results["cv_gradients"] = extracv(atoms)