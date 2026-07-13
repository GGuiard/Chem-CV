"""
Collect per-structure xyz results
"""

import argparse
from pathlib import Path
import subprocess

import ase.io


def gather_dft(
    name       = "dft",
    single_dir = "dft_single",
    outdir     = ".",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    single_dir = Path(single_dir)
    output     = Path(outdir) / f"{name}.xyz"

    files = sorted(single_dir.glob(f"{name}_*.xyz"))
    atoms_list = [ase.io.read(str(f)) for f in files]
    ase.io.write(str(output), atoms_list)

    # subprocess.run(f"rm -rf {str(single_dir)}", shell=True, check=True)


def _parser():
    p = argparse.ArgumentParser(description="Collect per-structure xyz results",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--name", default="dft", help="Single xyz files prefix")
    p.add_argument("--single-dir", default="dft_single", help="Directory with name_XXXX.xyz files")
    p.add_argument("--outdir", default=".", help="Output directory")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    
    gather_dft(
        name       = args.name,
        single_dir = args.single_dir,
        outdir     = args.outdir
    )