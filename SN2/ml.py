from mlcolvar.cvs import RegressionCV
from mlcolvar.data import DictModule
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torch import no_grad
from torch.optim import lr_scheduler
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from orca_parser import ChemCV

os.chdir("SN2")

subprocess.run("mkdir ML", shell=True)

# --- Import Data ---

# trajectories = read("DFT/sampling.xyz", ":")

# for atoms in trajectories:
#     atoms.center(1)
#     atoms.set_pbc(False)

# write("ML/trajectories.xyz", trajectories)

trajectories = read("ML/trajectories.xyz")

raw_chemcvs = ChemCV.load(path="DFT/CHEMCV", format="json").to_dict()
chemcvs = {
    "q_Hirshfeld": raw_chemcvs["q_Hirshfeld.5"] - raw_chemcvs["q_Hirshfeld.1"],
    "sigma_MBIS": raw_chemcvs["sigma_MBIS.5"] - raw_chemcvs["sigma_MBIS.1"],
    "b_Mayer": raw_chemcvs["b_Mayer.0.5"] - raw_chemcvs["b_Mayer.0.1"],
    "q_AO_Loewdin": raw_chemcvs["q_AO_Loewdin.5.p"] - raw_chemcvs["q_AO_Loewdin.1.p"],
    "p_MOAO_Loewdin": raw_chemcvs["p_MOAO_Loewdin.21.5.p"] - raw_chemcvs["p_MOAO_Loewdin.21.1.p"],
}

os.chdir("ML")

for chemcv_name, graph_labels in chemcvs.items():

    subprocess.run(f"mkdir {chemcv_name}", shell=True)
    np.save(f"{chemcv_name}/graph_labels", graph_labels)

    # --- Create Dataset ---
    dataset = create_dataset_from_trajectories(
        trajectories="trajectories.xyz",
        topologies=None,
        cutoff=6,
        graph_labels=[graph_labels.tolist()],
        node_labels=None,
        system_selection="type Cl",
        environment_selection="not type Cl",
    )

    datamodule = DictModule(dataset, shuffle=[True, False])

    # --- Create Model ---

    gnn_model = SchNetModel(
        n_out=1,
        cutoff=dataset.metadata["cutoff"],
        atomic_numbers=dataset.metadata["atomic_numbers"],
        pooling_operation="custom",
    )

    options = {
        'optimizer': {'lr': 1e-3},
        'lr_scheduler': {
            'scheduler': lr_scheduler.ExponentialLR,
            'gamma': 0.9999
        }
    }

    model = RegressionCV(gnn_model, options=options)

    # --- Train Model ---

    metrics = MetricsCallback()
    early_stopping = EarlyStopping(monitor="valid_loss", patience=20)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="valid_loss")

    trainer = Trainer(
        callbacks=[metrics, early_stopping, checkpoint],
        logger=True,
        enable_checkpointing=True,
        max_epochs=5000,
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule)

    # --- Make Plots ---

    ax = plot_metrics(
        metrics.metrics,          
        keys=['train_loss_epoch', 'valid_loss'],
        linestyles=['-.','-'],
        colors=['fessa1','fessa5'],
        yscale='log',
    )

    model.to_torchscript(f"{chemcv_name}/model.ptc", method="trace")
    plt.savefig(f"{chemcv_name}/plot_metrics.svg")


    model.eval()

    train_indices = datamodule._dataset_split[0].indices
    valid_indices = datamodule._dataset_split[1].indices

    ref_train = graph_labels[train_indices]
    ref_valid = graph_labels[valid_indices]

    with no_grad():
        pred = model(dataset.get_graph_inputs()).detach().squeeze()

    pred_train = pred[train_indices]
    pred_valid = pred[valid_indices]

    min_value = min(min(ref_valid), min(ref_train), min(pred_valid), min(pred_train))
    max_value = max(max(ref_valid), max(ref_train), max(pred_valid), max(pred_train))

    fig, ax = plt.subplots(layout='constrained')
    ax.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
    ax.scatter(ref_train, pred_train, s=2, label="train")
    ax.scatter(ref_valid, pred_valid, s=2, label="valid")

    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    ax.legend()
    ax.set_aspect('equal', 'box')

    fig.savefig(f"{chemcv_name}/pred_trainvalid.svg")