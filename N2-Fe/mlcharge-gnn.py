import os
os.chdir("N2-Fe/OPES_2")

from mlcolvar.cvs import RegressionCV
from mlcolvar.data import DictModule
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel

from lightning import Trainer
from torch import no_grad
from torch.optim import lr_scheduler

from ase.io import read

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

### Create Dataset ###

sampling_traj_with_traget = np.arange(0, 200001, 10)
rng = np.random.default_rng(42)
sampling = rng.choice(len(sampling_traj_with_traget), size=1000, replace=False)
np.random.shuffle(sampling)
sampling_traj = sampling_traj_with_traget[sampling]

trajectories = []
for i, index in enumerate(tqdm(sampling)):
    structure = read("traj_comp.traj", index)
    structure.set_pbc([True, True, True])
    trajectories.append(structure)

q = np.loadtxt("CHARGES")[sampling]
target = (q[:,72]+q[:,73])/2

dataset = create_dataset_from_trajectories(
    "traj.xyz",
    topologies=None,
    cutoff=4.5,
    graph_labels=target.tolist(),
    node_labels=q.tolist(),
    system_selection="type N",
    environment_selection="not type N",
)

datamodule = DictModule(dataset, batch_size=32)


### Create Model ###

gnn_model = SchNetModel(
    n_out=1,
    cutoff=dataset.metadata["cutoff"],
    atomic_numbers=dataset.metadata["atomic_numbers"],
)

options = {
    'optimizer': {'lr': 1e-3},
    'lr_scheduler': {
        'scheduler': lr_scheduler.ExponentialLR,
        'gamma': 0.9999
    }
}

model = RegressionCV(gnn_model, options=options)


### Train Model ###

metrics = MetricsCallback()

trainer = Trainer(
    callbacks=[metrics],
    logger=False,
    enable_checkpointing=False,
    max_epochs=10000,
    enable_model_summary=False,
)

trainer.fit(model, datamodule)
model.to_torchscript('../model.ptc', method="trace")

ax = plot_metrics(
    metrics.metrics,          
    keys=['train_loss_epoch','valid_loss'],
    linestyles=['-.','-'],
    colors=['fessa1','fessa5'],
    yscale='log',
)
plt.savefig('plot_metrics.svg')

### Plots ###

model.eval()

train_indices = datamodule._dataset_split[0].indices
valid_indices = datamodule._dataset_split[1].indices

ref_train = target[train_indices]
ref_valid = target[valid_indices]

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

fig.savefig("pred.svg")
