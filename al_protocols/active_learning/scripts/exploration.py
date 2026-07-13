"""
Biased MD with OPES-Explore (Bussi thermostat)
"""

import argparse
import os
from pathlib import Path

import ase.io
import ase.units
from ase.calculators.plumed import Plumed
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm


def run_md(
    timestep      = 0.5,    # fs
    run_time      = 1000.0, # ps
    temperature   = 300.0,  # K
    taut          = 100.0,  # fs
    interval_traj = 100,
    path_init     = "init.xyz",
    path_plumed   = "plumed.dat",
    path_franken  = None,
    outdir        = ".",
):
    prevdir = os.getcwd()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    path_init    = prevdir / Path(path_init)
    path_plumed  = prevdir / Path(path_plumed)
    if path_franken is not None:
        path_franken = prevdir / Path(path_franken)

    nb_steps = int(run_time * 1e3 // timestep)

    # Get initial configuration
    atoms = ase.io.read(str(path_init))

    # Get MLIP calculator
    if path_franken is not None:
        from franken.calculators import FrankenCalculator
        base_calc = FrankenCalculator(str(path_franken))
    else:
        from mace.calculators import mace_mp
        base_calc = mace_mp(model="mh-1", head="spice_wB97M")

    # Get PLUMED bias
    plumed_input = open(str(path_plumed)).read().splitlines()
    atoms.calc = Plumed(base_calc, plumed_input, timestep * ase.units.fs,
                        atoms, ase.units.kB * temperature)

    # Add Bussi thermostat
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Bussi(atoms, timestep * ase.units.fs, temperature, taut * ase.units.fs)

    # Save trajectory
    traj = ase.io.Trajectory("traj_comp.traj", "w", atoms)
    dyn.attach(traj, interval_traj)

    # Add progress bar
    pbar = tqdm(total=100, unit="%", desc="MD")
    _first = [True]
    def _tick():
        if _first[0]: _first[0] = False
        else: pbar.update(1)
    dyn.attach(_tick, max(1, nb_steps // 100))

    # Run the simulation
    dyn.run(nb_steps)

    pbar.close()
    os.chdir(prevdir)


def _parser():
    p = argparse.ArgumentParser(description="Biased MD with OPES-Explore (Bussi thermostat)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--timestep", type=float, default=0.5, help="Timestep [fs]")
    p.add_argument("--run-time", type=float, default=1000.0, help="Run time [ps]")
    p.add_argument("--temperature", type=float, default=300.0, help="Temperature [K]")
    p.add_argument("--taut", type=float, default=100.0, help="Bussi coupling [fs]")
    p.add_argument("--interval-traj", type=int, default=100, help="Save every N steps")
    p.add_argument("--init", default="init.xyz", help="Initial structure (.xyz)")
    p.add_argument("--plumed", default="plumed.dat", help="PLUMED input file")
    p.add_argument("--franken", default=None, help="Franken model (.pt); uses mace_mp if absent")
    p.add_argument("--outdir", default=".", help="Output directory")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    run_md(
        path_init     = args.init,
        path_plumed   = args.plumed,
        outdir        = args.outdir,
        path_franken  = args.franken,
        run_time      = args.run_time,
        temperature   = args.temperature,
        taut          = args.taut,
        timestep      = args.timestep,
        interval_traj = args.interval_traj,
    )
