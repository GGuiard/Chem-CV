import os
os.chdir("N2-Fe/OPES_2")

import numpy as np
from mlcolvar.data import DictModule
from mlcolvar.data.utils import save_dataset, load_dataset
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import no_grad
from torch.jit import load
from ase.io import read, write
import matplotlib.pyplot as plt
from rich.progress import Progress

use_dataset = False
use_model = False

### Create Dataset ###

if use_dataset:
    dataset = load_dataset("dataset")

else:
    sampling = np.arange(100) #20000
    sampling_traj_with_traget = np.arange(0, 200001, 10)
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

    dataset = create_dataset_from_trajectories("traj.xyz",
                                               topologies=None,
                                               cutoff=4.0,
                                               graph_labels=target,
                                               system_selection="type N",
                                               environment_selection="not type N")

    save_dataset(dataset, "dataset")

### Create DataModule ###

datamodule = DictModule(dataset, lengths=[0.8,0.2])

### Create and Train Model ###

if use_model:
    model = load('../model.ptc')

else:
    gnn_model = SchNetModel(n_out=1,
                            cutoff=dataset.metadata["cutoff"],
                            atomic_numbers=dataset.metadata["atomic_numbers"],
                            w_out_after_pool=False,
                            n_layers=2)

    options = {'optimizer': {'lr': 1e-3}}
    model = RegressionCV(gnn_model, options=options)

    metrics = MetricsCallback()
    early_stopping = EarlyStopping(monitor="valid_loss",
                                   patience=5000,
                                   min_delta=1e-5)
   
    trainer = Trainer(callbacks=[metrics, early_stopping],
                      enable_model_summary=False,
                      max_epochs=1000)

    trainer.fit(model, datamodule)

    ax = plot_metrics(metrics.metrics,
                      keys=['train_loss_epoch','valid_loss'],
                      linestyles=['-.','-'],
                      colors=['fessa1','fessa5'],
                      yscale='log')
    plt.savefig('plot_metrics.svg')

    model.to_torchscript('../model.ptc', method="trace")

### Plots ###

datamodule.setup()
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