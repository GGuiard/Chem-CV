"""
md_runner.py
============
A reusable module for running ASE-based molecular dynamics simulations
with optional PLUMED enhanced sampling and the Bussi thermostat.

Typical usage
-------------
From any directory containing your system files:

    from md_runner import md_run

    md_run(
        calc="off",
        plumed_file="plumed-opes.dat",
    )
"""

import subprocess
from pathlib import Path

from tqdm import tqdm

from ase import units
from ase.calculators.plumed import Plumed, restart_from_trajectory
from ase.io import read, Trajectory
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def md_run(
    # --- System ---
    init_file: str = "init.xyz",
    calc = None,
    # --- Thermostat ---
    T: float = 300.0,           # K
    taut: float = 100.0,        # fs  (Bussi relaxation time)
    # --- Simulation time ---
    timestep: float = 0.5,            # fs
    total_time: float = 1_000_000.0,  # fs
    # --- PLUMED ---
    plumed_file: str | None = None,
    # --- Output control ---
    interval_info: int | None = None,  # steps; default = total_steps // 100
    interval_traj: int = 10,           # steps; must be a multiple of plumed stride
    save_energy: bool = True,
    save_traj: bool = True,
    use_progress_bar: bool = True,
    # --- Restart ---
    restart: bool = False,
    prev_step: int = 0,
) -> None:
    """Run (or restart) a Bussi-thermostatted MD simulation.

    Parameters
    ----------
    init_file:
        Path to the initial structure file (any ASE-readable format).
        Only used when *restart* is False.
    calc:
        An ASE calculator instance.  Required — no default is provided
        so that the caller explicitly chooses the level of theory.
    T:
        Target temperature in Kelvin.
    taut:
        Bussi thermostat relaxation time in femtoseconds.
    timestep:
        Integration time step in femtoseconds.
    total_time:
        Total simulation wall-clock time in femtoseconds.
        When restarting, *prev_step* steps are subtracted automatically.
    plumed_file:
        Path to a PLUMED input file.  Pass ``None`` to run plain MD.
    interval_info:
        How often (in steps) to write the ENERGY file.
        Defaults to ``total_steps // 100`` (i.e. 100 data points total).
    interval_traj:
        How often (in steps) to append a frame to the trajectory.
        Must be a multiple of the PLUMED STRIDE if PLUMED is used.
    save_energy:
        Whether to write a plain-text ENERGY file.
    save_traj:
        Whether to write a binary ASE trajectory file.
    use_progress_bar:
        Whether to display a tqdm progress bar in the terminal.
    restart:
        Set to True to append to an existing ``traj_comp.traj`` and resume
        a previous PLUMED run (using ``restart_from_trajectory``).
    prev_step:
        Number of steps already completed in the trajectory being restarted.
        Only used when *restart* is True.

    Raises
    ------
    ValueError
        If *calc* is not provided.
    FileNotFoundError
        If *init_file* (fresh run) or ``traj_comp.traj`` (restart) is missing.
    """

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    if calc is None:
        raise ValueError(
            "A calculator must be supplied via the `calc` argument. "
            "Example: calc=mace_off(model='...', default_dtype='float32')"
        )
    
    elif calc == 'mh':
        from mace.calculators import mace_mp
        calc = mace_mp(
            model='mh-0',
            head='oc20_usemppbe',
            )
        
    elif calc == 'off':
        from mace.calculators import mace_off
        calc = mace_off(
            model="https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off24/MACE-OFF24_medium.model",
            default_dtype="float32",
            )

    # -----------------------------------------------------------------------
    # Derived parameters
    # -----------------------------------------------------------------------
    kT = units.kB * T
    nb_steps = int(total_time // timestep)
    if interval_info is None:
        interval_info = max(1, nb_steps // 100)
    if restart:
        nb_steps -= prev_step

    # -----------------------------------------------------------------------
    # House-keeping: clean previous output files on a fresh run
    # -----------------------------------------------------------------------
    if not restart:
        subprocess.run(
            "rm -f bck.* *.traj COLVAR KERNELS STATES ENERGY",
            shell=True,
        )

    # -----------------------------------------------------------------------
    # Load structure
    # -----------------------------------------------------------------------
    if not restart:
        if not Path(init_file).exists():
            raise FileNotFoundError(f"Initial structure file not found: {init_file}")
        atoms = read(init_file)
        
    else:
        traj_path = Path("traj_comp.traj")
        if not traj_path.exists():
            raise FileNotFoundError(
                "Restart requested but 'traj_comp.traj' was not found in the "
                "current directory."
            )
        atoms = read(str(traj_path))

    # -----------------------------------------------------------------------
    # Attach calculator (plain or PLUMED-wrapped)
    # -----------------------------------------------------------------------
    if plumed_file:
        plumed_input = Path(plumed_file).read_text().splitlines()
        if not restart:
            plumed_calc = Plumed(
                calc=calc,
                input=plumed_input,
                timestep=timestep * units.fs,
                atoms=atoms,
                kT=kT,
            )
        else:
            plumed_calc = restart_from_trajectory(
                prev_traj="traj_comp.traj",
                prev_steps=prev_step,
                calc=calc,
                input=plumed_input,
                timestep=timestep * units.fs,
                atoms=atoms,
                kT=kT,
            )
        atoms.calc = plumed_calc
    else:
        atoms.calc = calc

    # -----------------------------------------------------------------------
    # Initialise velocities and create the Bussi propagator
    # -----------------------------------------------------------------------
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    dyn = Bussi(atoms, timestep * units.fs, T, taut * units.fs)

    # -----------------------------------------------------------------------
    # Attach: energy / temperature logger
    # -----------------------------------------------------------------------
    if save_energy:
        # Write header on fresh runs only
        if not restart:
            with open("ENERGY", "w") as f:
                f.write("#! FIELDS Emec Temp\n")

        def _write_energy():
            E_mec = atoms.calc.results["energy"][0] + atoms.get_kinetic_energy()
            temp  = atoms.get_temperature()
            with open("ENERGY", "a") as f:
                f.write(f"{E_mec:9.6f} {temp:9.6f}\n")

        dyn.attach(_write_energy, interval_info)

    # -----------------------------------------------------------------------
    # Attach: trajectory writer
    # -----------------------------------------------------------------------
    if save_traj:
        if not restart:
            traj = Trajectory("traj_comp.traj", 'w', atoms)
        else:
            traj = Trajectory("traj_comp.traj", 'a')
        dyn.attach(traj, interval_traj)

    # -----------------------------------------------------------------------
    # Attach: progress bar (100 ticks over the whole run)
    # -----------------------------------------------------------------------
    if use_progress_bar:
        pbar = tqdm(total=100, unit="%", desc="MD progress")
        tick_interval = max(1, nb_steps // 100)

        # ASE calls all attached functions once at step 0
        # which would corrupt tqdm's ETA estimate.
        _pbar_initialised = False
 
        def _update_pbar():
            nonlocal _pbar_initialised
            if not _pbar_initialised:
                _pbar_initialised = True
                return
            pbar.update()


        dyn.attach(_update_pbar, tick_interval)

    # -----------------------------------------------------------------------
    # Run the dynamics
    # -----------------------------------------------------------------------
    try:
        dyn.run(nb_steps)
    finally:
        # Guarantee the progress bar is always closed, even on error
        if use_progress_bar:
            pbar.close()
        # Flush the trajectory to disk
        if save_traj:
            traj.close()
