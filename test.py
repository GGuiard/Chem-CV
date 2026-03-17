import subprocess

subprocess.run("which plumed", shell=True)
subprocess.run("plumed config module opes", shell=True)
subprocess.run("echo $PLUMED_KERNEL", shell=True)
subprocess.run("which gmx_mpi", shell=True)

from ase import Atoms
from ase.build import molecule
from ase.calculators.plumed import Plumed
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen  # fallback if Bussi not available
from ase.md.nvtbussi import NVTBussi
from ase import units

from mace.calculators import MACECalculator

# -----------------------
# 1. Build toy system
# -----------------------
atoms = molecule("H2O")  # simple molecule
atoms.center(vacuum=5.0)

# -----------------------
# 2. MACE calculator
# -----------------------
calc = MACECalculator(
    model_path="your_model.model",  # <-- replace with your model
    device="cpu"  # or "cuda"
)

# -----------------------
# 3. PLUMED input (OPES)
# -----------------------
plumed_input = """
# Distance between atoms 1 and 2
d: DISTANCE ATOMS=1,2

# OPES bias
opes: OPES_METAD ARG=d \
    PACE=200 \
    BARRIER=20 \
    TEMP=300

# Output
PRINT STRIDE=10 ARG=d,opes.bias FILE=COLVAR
"""

# -----------------------
# 4. Wrap calculator with PLUMED
# -----------------------
plumed_calc = Plumed(
    calc=calc,
    input=plumed_input,
    timestep=1.0 * units.fs,
    atoms=atoms
)

atoms.calc = plumed_calc

# -----------------------
# 5. Initialize velocities
# -----------------------
temperature = 300  # K
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# -----------------------
# 6. NVT dynamics (Bussi thermostat)
# -----------------------
timestep = 1.0 * units.fs

dyn = NVTBussi(
    atoms,
    timestep,
    temperature_K=temperature,
    taut=100 * units.fs  # thermostat time constant
)

# -----------------------
# 7. Run dynamics
# -----------------------
def print_status(a=atoms):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * len(a) * units.kB)
    print(f"Step energy: Epot={epot:.3f} eV, Temp={temp:.1f} K")

dyn.attach(print_status, interval=50)

print("Starting MD...")
dyn.run(2000)

print("Done.")