"""
Franken fine-tuning
"""

import argparse
from pathlib import Path

from franken.config import MaceBackboneConfig, GaussianRFConfig
from franken.data import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from franken.trainers.rf_cuda_lowmem import RandomFeaturesTrainer
import torch


def run_training(
    backbone_id  = "mace_mh/1",
    inter_block  = 2,
    num_rf       = 2048,
    length_scale = 30.0,
    l2_penalty   = [1e-5],
    force_weight = [0.9],
    path_train   = "train.xyz",
    outdir       = ".",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gnn_config = MaceBackboneConfig(
        path_or_id        = backbone_id,
        interaction_block = inter_block,
    )

    train_dset = FrankenAtomsDataset(
        data_path  = path_train,
        split      = "train",
        gnn_config = gnn_config,
    )
    train_dl = train_dset.get_dataloader(distributed=False)

    rf_config  = GaussianRFConfig(
        num_random_features = num_rf,
        length_scale        = length_scale
    )

    model = FrankenPotential(
        gnn_config     = gnn_config,
        rf_config      = rf_config,
        num_species    = 3, 
        jac_chunk_size = "auto",
    )

    trainer = RandomFeaturesTrainer(
        train_dataloader = train_dl,
        save_every_model = False,
        device           = device,
        save_fmaps       = False,
    )

    solver_params = {
        "l2_penalty":   l2_penalty,
        "force_weight": force_weight
    }

    _, weights = trainer.fit(model, solver_params)

    model.rf.weights = weights
    model.save(str(outdir / "franken.pt"))

    return weights


import ase.io
from franken.calculators import FrankenCalculator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_parity(
    path_model = "franken.pt",
    path_test  = "test.xyz",
    outdir     = ".",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calc = FrankenCalculator(path_model)

    test_dset = ase.io.read(path_test, ':')

    targets_e, predictions_e = [], []
    targets_f, predictions_f = [], []

    for atoms in tqdm(test_dset, desc="Evaluating"):
        targets_e.append(float(atoms.get_potential_energy()))
        targets_f.append(atoms.get_forces())

        atoms.calc = calc
        predictions_e.append(float(atoms.get_potential_energy()))
        predictions_f.append(atoms.get_forces())

    targets_e      = np.array(targets_e)
    predictions_e  = np.array(predictions_e)
    targets_e     -= targets_e.mean()
    predictions_e -= predictions_e.mean()
    e_mae = np.mean(np.abs(targets_e - predictions_e)) * 1e3
    f_mae = np.mean(np.abs(np.array(targets_f) - np.array(predictions_f))) * 1e3

    bounds = [min(targets_e.min(), predictions_e.min()), max(targets_e.max(), predictions_e.max())]
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(targets_e, predictions_e, s=4.)
    ax.plot(bounds, bounds, c="k", ls="--")
    ax.text(bounds[0], 0.95 * bounds[1],
            f"Energy MAE = {e_mae:.2f} meV\nForces MAE = {f_mae:.2f} meV/Å")
    ax.set_xlabel("Energy Reference [eV]")
    ax.set_ylabel("Energy Prediction [eV]")
    ax.set_aspect("equal", "box")

    out = outdir / "parity.png"
    fig.savefig(str(out))


def _parser():
    p = argparse.ArgumentParser(description="Fine-tune FrankenPotential",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--backbone", default="mace_mh/1", help="GNN backbone ID")
    p.add_argument("--num-rf", type=int, default=2048, help="Random features")
    p.add_argument("--length-scale", type=float, default=30.0, help="RF length scale")
    p.add_argument("--l2-penalty", type=float, default=5, help="Trainer L2 penalty in 10^{-X}")
    p.add_argument("--force-weight", type=float, default=0.9, help="Trainer force weight")
    p.add_argument("--train", default="train.xyz", help="Train xyz")
    p.add_argument("--test", default="test.xyz", help="Test xyz")
    p.add_argument("--outdir", default=".", help="Output directory")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    args.l2_penalty = [10**(-args.l2_penalty)]
    args.force_weight = [args.force_weight]

    run_training(
        backbone_id  = args.backbone,
        num_rf       = args.num_rf,
        length_scale = args.length_scale,
        l2_penalty   = args.l2_penalty,
        force_weight = args.force_weight,
        path_train   = args.train,
        outdir       = args.outdir,
    )

    plot_parity(
        path_model = args.outdir + "/franken.pt",
        path_test  = args.test,
        outdir     = args.outdir,
    )