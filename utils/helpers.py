"""
helpers.py
==========
General helpers functions for molecular dynamics.
"""

import numpy as np
from ase.io import read, write
from scipy.spatial.transform import Rotation
import subprocess

# ---------------------------------------------------------------------------
# Atoms object modification
# ---------------------------------------------------------------------------

def orient_atoms(
    atoms,
    center: int = 0,
    x_axis: int = 1,
    y_axis: int = 2
):
    atoms = atoms.copy()

    # --- 1. Move atom 0 to center of cell ---
    cell_center = atoms.cell.diagonal() / 2.0
    shift = cell_center - atoms.positions[center]
    atoms.positions += shift

    # vectors from atom 0
    rx = atoms.positions[x_axis] - atoms.positions[center]
    ry = atoms.positions[y_axis] - atoms.positions[center]

    # --- 2. Rotate r01 onto x-axis ---
    ex = np.array([1.0, 0.0, 0.0])

    rxn = rx / np.linalg.norm(rx)

    axis = np.cross(rxn, ex)
    angle = np.arccos(np.clip(np.dot(rxn, ex), -1.0, 1.0))

    if np.linalg.norm(axis) > 1e-12:
        axis /= np.linalg.norm(axis)
        rot1 = Rotation.from_rotvec(angle * axis)
        atoms.positions = rot1.apply(atoms.positions - atoms.positions[center]) + atoms.positions[center]

    # recompute after first rotation
    ry = atoms.positions[y_axis] - atoms.positions[center]

    # --- 3. Rotate around x-axis so atom 2 lies in xy-plane ---
    yz_angle = np.arctan2(ry[2], ry[1])

    rot2 = Rotation.from_rotvec(-yz_angle * ex)
    atoms.positions = rot2.apply(atoms.positions - atoms.positions[center]) + atoms.positions[center]

    return atoms


def remove_com_atoms(atoms):
    atoms.positions -= atoms.get_center_of_mass()
    return atoms


# ---------------------------------------------------------------------------
# Trajectory file modification
# ---------------------------------------------------------------------------

def iloc_traj(traj: str, indices: np.ndarray):
    return [read(traj, index) for index in indices]


def orient_traj(
    traj,
    center: int = 0,
    x_axis: int = 1,
    y_axis: int = 2
):
    return [orient_atoms(atoms, center, x_axis, y_axis) for atoms in traj]


def remove_com_traj(traj):
    return [remove_com_atoms(atoms) for atoms in traj]


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def bin_sampling(
    cv: np.ndarray,
    nb_bins: int = 20,
    nb_per_bin: int = 50,
    bounds: tuple = (None, None),
) -> np.ndarray:
    if not bounds[0]:
        bounds[0] = np.min(cv)
    if not bounds[1]:
        bounds[1] = np.max(cv)

    bins = np.linspace(bounds[0], bounds[1], nb_bins+1)

    indices_per_bin = []
    for bin in zip(bins[:-1], bins[1:]):
        arg = np.argwhere(np.logical_and(bin[0]<cv, cv<bin[1])).T[0]
        sample = np.random.choice(arg, nb_per_bin, replace=False)
        indices_per_bin.append(sample)

    return np.array(indices_per_bin).ravel()


def split_traj(
    traj: str,
    nb_split: int,
    directory: str,
) -> None:
    traj = read(traj, ":")
    nb_traj = len(traj)

    splits = np.array_split(np.arange(nb_traj), nb_split)

    for i, indices_split in enumerate(splits):
        sampling_split = read(traj, f"{indices_split[0]}:{indices_split[-1]+1}")

        subprocess.run(f"mkdir {directory}/split_{i}", shell=True)
        write(f"{directory}/split_{i}/{traj}", sampling_split, format="xyz")


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def bin_2D(
    x: np.ndarray,
    y: np.ndarray,
    nb_bins: int = 20,
    bounds: tuple = (None, None),
) -> np.ndarray:
    if not bounds[0]:
        bounds[0] = np.min(x)
    if not bounds[1]:
        bounds[1] = np.max(x)

    bins = np.linspace(bounds[0], bounds[1], nb_bins+1)

    dataset = []
    for bin in zip(bins[:-1], bins[1:]):
        arg = np.argwhere(np.logical_and(bin[0]<x, x<bin[1])).T[0]
        dataset.append(y[arg])

    return bins, dataset