import os
os.chdir("N2-Fe/OPES_2")

import numpy as np
from mlcolvar.data import DictModule
# from mlcolvar.data.utils import save_dataset_configurations_as_extyz
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel
from lightning import Trainer
from torch import no_grad
from torch.optim import lr_scheduler
from ase.io import read, write
import matplotlib.pyplot as plt
from rich.progress import Progress

### Create Dataset ###

sampling_traj_with_traget = np.arange(0, 200001, 10)
# rng = np.random.default_rng(42)
# sampling = rng.choice(len(sampling_traj_with_traget), size=100, replace=False)
sampling = np.arange(100)
np.random.shuffle(sampling)
sampling_traj = sampling_traj_with_traget[sampling]

progress = Progress()
task = progress.add_task("Processing...", total=len(sampling))
trajectories = [None for i in range(len(sampling))]
progress.start()
for i, index in enumerate(sampling):
    trajectories[i] = read("traj_comp.traj", index)
    trajectories[i].set_pbc([True, True, True])
    progress.update(task, advance=1)
progress.stop()
write('traj.xyz', trajectories)

q = np.loadtxt("CHARGES")[sampling]
target = (q[:,72]+q[:,73])/2
target_mean = target.mean()
target_std  = target.std()
target = (target - target_mean) / target_std

dataset = create_dataset_from_trajectories(
    "traj.xyz",
    topologies=None,
    cutoff=4.0,
    graph_labels=target,
    system_selection="type N",
    subsystem_selection="type N",
    long_range_cutoff=10,
    environment_selection="not type N",
    )

datamodule = DictModule(dataset)

# save_dataset_configurations_as_extyz(dataset, "dataset.xyz")

### Create and Train Model ###

gnn_model = SchNetModel(
    n_out=1,
    cutoff=dataset.metadata["cutoff"],
    atomic_numbers=dataset.metadata["atomic_numbers"],
    w_out_after_pool=True,
    n_layers=2,
    )

options = {
    'optimizer': {'lr': 1e-3},
    'lr_scheduler': {
        'scheduler': lr_scheduler.ExponentialLR,
        'gamma': 0.9999
        }
    }
model = RegressionCV(gnn_model, options=options)

metrics = MetricsCallback()

trainer = Trainer(
    callbacks=[metrics],
    logger=False,
    enable_checkpointing=False,
    max_epochs=1000,
    enable_model_summary=False,
    )

trainer.fit(model, datamodule)

ax = plot_metrics(
    metrics.metrics,          
    keys=['train_loss_epoch','valid_loss'],
    linestyles=['-.','-'],
    colors=['fessa1','fessa5'],
    yscale='log',
    )
plt.savefig('plot_metrics.svg')

model.to_torchscript('../model.ptc', method="trace")

### Plots ###

train_indices = datamodule._dataset_split[0].indices
val_indices = datamodule._dataset_split[1].indices

ref_dataset = np.array([data['graph_labels'].item() for data in dataset['data_list']])
ref_train = ref_dataset[train_indices]
ref_valid = ref_dataset[val_indices]

model.eval()
with no_grad():
    pred_train = model(datamodule.get_graph_inputs("train")).detach().squeeze()
    pred_valid = model(datamodule.get_graph_inputs("valid")).detach().squeeze()

min_value = min(min(ref_valid), min(ref_train), min(pred_valid), min(pred_train))
max_value = max(max(ref_valid), max(ref_train), max(pred_valid), max(pred_train))

fig, ax = plt.subplots(layout='tight')
ax.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
ax.plot(ref_train, pred_train, 'o', ms=1, label="train")
ax.plot(ref_valid, pred_valid, 'o', ms=1, label="valid")

ax.set_xlabel("Reference")
ax.set_ylabel("Prediction")
ax.legend()
ax.set_aspect('equal', 'box')

fig.savefig("pred.svg")