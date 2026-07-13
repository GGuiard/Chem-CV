"""
Configuration selection with DEAL
"""

import argparse
import os
from pathlib import Path

import ase.io
from deal import DataConfig, DEALConfig, FlareConfig, DEAL
import numpy as np


def _single_run(path_traj, threshold, lo, hi, target, max_selected):
    deal = DEAL(DataConfig(files=str(path_traj), shuffle=True), DEALConfig(threshold), FlareConfig())
    deal.run()
    n = len(ase.io.read("deal_selected.xyz", ":"))
    if n < target: hi = threshold
    elif n > max_selected: lo = threshold
    print(f"[DEAL] selected={n} threshold={threshold:.4f} bounds=({lo:.4f}, {hi:.4f})")
    return n, lo, hi


def run_deal(
    target    = 1000,
    max_extra = 0.1,
    threshold = 0.5,
    path_traj = "traj_comp.traj",
    outdir    = ".",
):
    prevdir = os.getcwd()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)

    path_traj = prevdir / Path(path_traj)

    max_selected = target * (1 + max_extra)

    # Initial run
    n, lo, hi = _single_run(path_traj, threshold, 0, 1, target, max_selected)

    # Exponential decrease to get lower bound
    print("[DEAL] exponential decrease")
    while n < target and not (target <= n <= max_selected):
        threshold *= 0.5
        n, lo, hi = _single_run(path_traj, threshold, lo, hi, target, max_selected)

    # Bisection to land in acceptable region
    print("[DEAL] bisection")
    while not (target <= n <= max_selected):
        threshold = np.sqrt(lo * hi)
        n, lo, hi = _single_run(path_traj, threshold, lo, hi, target, max_selected)

    # Trim to the exact number of configuration wished
    selected = ase.io.read("deal_selected.xyz", ":")
    chosen = np.sort(np.random.choice(len(selected), target, replace=False))
    selected = [selected[i] for i in chosen]
    selected = sorted(selected, key=lambda atom: atom.info["original_frame"])
    ase.io.write("deal_selected.xyz", selected)

    os.chdir(prevdir)


def _parser():
    p = argparse.ArgumentParser(description="DEAL configuration selection",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--target", type=int, default=1000, help="Target number of configs")
    p.add_argument("--max-extra", type=int, default=0.1, help="Allowed overshoot above target")
    p.add_argument("--threshold", type=float, default=0.5, help="Initial DEAL threshold")
    p.add_argument("--traj", default="traj_comp.traj", help="Input trajectory")
    p.add_argument("--outdir", default=".", help="Output directory")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    run_deal(
        path_traj = args.traj,
        outdir    = args.outdir,
        target    = args.target,
        max_extra = args.max_extra,
        threshold = args.threshold,
    )