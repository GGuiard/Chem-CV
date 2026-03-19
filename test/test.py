import plumed

from ase import Atoms
from ase.build import molecule
from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.bussi import Bussi
from ase import units

from mace.calculators import mace_mp

# Simulation parameters
temperature = 300
timestep = units.fs

# Setup system
atoms = molecule("H2O")
atoms.center(vacuum=5.0)

# Setup MACE calculator
calc = mace_mp(model="small")

# Setup PLUMED OPES
setup = ["UNITS LENGTH=A", "d: DISTANCE ATOMS=1,2", f"opes: OPES_METAD ARG=d PACE=200 BARRIER=20 TEMP={temperature}", "PRINT STRIDE=10 ARG=d,opes.bias FILE=COLVAR", "FLUSH STRIDE=10"]
plumed_calc = Plumed(calc=calc, input=setup, timestep=timestep, atoms=atoms)
atoms.calc = plumed_calc

# Setup Bussi propagator
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
dyn = Bussi(atoms, timestep=timestep, temperature_K=temperature, taut=100*timestep)

# Extract useful quantities
def print_status(a=atoms):
    epot = a.get_potential_energy()[0]
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * len(a) * units.kB)
    print(f"Epot = {epot:.3f}\nEmec = {epot+ekin:.3f}\nTemp = {temp:.1f}\n")
dyn.attach(print_status, interval=50)

# Run simulation
dyn.run(2000)
