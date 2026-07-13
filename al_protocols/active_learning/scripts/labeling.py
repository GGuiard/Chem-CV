"""
ORCA DFT on one structure
"""

import argparse
import contextlib
import os
import subprocess
from pathlib import Path

import ase.io
from ase.calculators.orca import ORCA
from orca_utils import AVAILABLE_CHEMCVS


def run_dft_single(
    index,
    charge     = 0,
    mult       = 1,
    theory     = "WB97X-D4",
    basis      = "def2-TZVP",
    bonds      = [(0, 1)],
    nprocs     = 8,
    path_input = "deal_selected.xyz",
    outdir     = "dft_single",
    force      = False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = outdir / f"dft_{index:04d}.xyz"
    if out.exists() and not force:
        return

    atoms = ase.io.read(str(path_input), index=index)

    chemcv = AVAILABLE_CHEMCVS["b_Mayer"]
    simpleinput     = chemcv["simpleinput"]
    block           = chemcv["block"]
    source          = chemcv["source"]
    parsingfunction = chemcv["parsingfunction"]

    scratch = outdir / f"orca_{index:04d}"
    scratch.mkdir(exist_ok=True)

    atoms.calc = ORCA(
        charge          = charge,
        mult            = mult,
        directory       = str(scratch),
        orcasimpleinput = f"{theory} {basis} EnGrad {simpleinput}",
        orcablocks      = f"%pal nprocs {nprocs} end\n{block}",
    )

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            atoms.get_potential_energy()

    props = source(str(scratch))
    cvs   = parsingfunction(props, bonds=bonds)
    for bond in enumerate(bonds):
        atoms.info[f"bond-{bond[0]}-{bond[1]}"] = cvs.get(bond, 0.0)

    ase.io.write(str(out), atoms)
    subprocess.run(f"rm -rf {scratch}", shell=True, check=True)

def bonds_type(arg):
    return [tuple(map(int, bond.split('-'))) for bond in str(arg).split(',')]

def _parser():
    p = argparse.ArgumentParser(description="ORCA DFT on one structure",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--index", type=int, default=0, help="Structure index")
    p.add_argument("--charge", type=int, default=0, help="Molecular charge")
    p.add_argument("--mult", type=int, default=1, help="Spin multiplicity")
    p.add_argument("--theory", default="B3LYP", help="ORCA DFT functional")
    p.add_argument("--basis", default="def2-TZVP", help="ORCA basis")
    p.add_argument("--bonds", default="0-1", type=bonds_type, help="Mayer bonds extracted (format 0-1,0-2)")
    p.add_argument("--nprocs", type=int, default=8, help="ORCA OMP threads")
    p.add_argument("--input", default="deal_selected.xyz", help="Input xyz")
    p.add_argument("--outdir", default="dft_single", help="Output directory")
    p.add_argument("--force", action="store_true", help="Redo the calculation even if output already exists")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    run_dft_single(
        index      = args.index,
        charge     = args.charge,
        mult       = args.mult,
        theory     = args.theory,
        basis      = args.basis,
        nprocs     = args.nprocs,
        path_input = args.input,
        outdir     = args.outdir,
        force      = args.force,
    )