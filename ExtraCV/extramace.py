from mace.calculators import MACECalculator
from ase.calculators.calculator import all_changes

class MyCalculator(MACECalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "extracv" in kwargs:
            self.extracv = kwargs["extracv"]
            
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.extracv is not None:
            extra_cv, extra_cv_gradients = {}, {}
            for name, model in self.extracv.items():
                extra_cv[name], extra_cv_gradients[name] = model(atoms)
            self.results["cv"], self.results["cv_gradients"] = extra_cv, extra_cv_gradients