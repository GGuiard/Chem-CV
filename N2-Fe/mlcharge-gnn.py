import os
os.chdir("N2-Fe/OPES_2")

import numpy as np
from mlcolvar.data import DictModule
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_configurations
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.data.graph.atomic import Configuration, AtomicNumberTable
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ase.io import read
import matplotlib.pyplot as plt

### Import data ###

sampling_start, sampling_end, sampling_stride = 0, 2001, 10

q = np.loadtxt("CHARGES")[:201]

### Create Dataset ###

trajectories = read("traj_comp.traj", f"{sampling_start}:{sampling_end}:{sampling_stride}")
atomic_numbers = trajectories[0].get_atomic_numbers()
cell = np.array(trajectories[0].get_cell())
pbc = trajectories[0].get_pbc()
z_table = AtomicNumberTable.from_zs(atomic_numbers)

configurations = []
for i in range(len(trajectories)):
    configuration = Configuration(atomic_numbers, trajectories[i].get_positions(), cell, pbc, None, None)    
    configurations.append(configuration)

dataset = create_dataset_from_configurations(configurations, z_table, cutoff=10.0, show_progress=False)
for i, data_list in enumerate(dataset["data_list"]):
    data_list["graph_labels"] = i

target = Tensor((q[:,72]+q[:,73])/2)
dataset["target"] = target

datamodule = DictModule(dataset, lengths=[0.8,0.2])

### Create Model ###

model = SchNetModel(n_out=1, cutoff=dataset.metadata["cutoff"], atomic_numbers=dataset.metadata["atomic_numbers"], w_out_after_pool=False)
model = RegressionCV(model)

### Define Trainer ###

metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", patience=10, min_delta=1e-5)

trainer = Trainer(callbacks=[metrics, early_stopping], enable_model_summary=False)

### Optimization ###

trainer.fit(model, datamodule)

### Plots ###

ax = plot_metrics(metrics.metrics, keys=['train_loss_epoch','valid_loss'], linestyles=['-.','-'], colors=['fessa1','fessa5'], yscale='log')

plt.show()

### Compile ###

# model.to_torchscript('model.ptc', method="trace")