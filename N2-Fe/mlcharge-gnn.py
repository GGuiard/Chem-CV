import os
os.chdir("N2-Fe")
import figures
os.chdir("OPES_2")

import numpy as np
from mlcolvar.data import DictModule
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_configurations
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.data.graph.atomic import Configuration, AtomicNumberTable
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ase.io import read
import matplotlib.pyplot as plt
from rich.progress import Progress

### Import data ###

sampling = np.arange(0, 100, 1)
sampling_traj = np.arange(0, 200001, 10)[sampling]

progress = Progress()
task = progress.add_task("Processing...", total=len(sampling))
trajectories = [None for i in range(len(sampling))]
progress.start()
for i, index in enumerate(sampling):
    trajectories[i] = read("traj_comp.traj", index)
    progress.update(task, advance=1)
progress.stop()

q = np.loadtxt("CHARGES")[sampling]

### Create Dataset ###

atomic_numbers = trajectories[0].get_atomic_numbers()
cell = np.array(trajectories[0].get_cell())
pbc = trajectories[0].get_pbc()
z_table = AtomicNumberTable.from_zs(atomic_numbers)

target = (q[:,72]+q[:,73])/2

configurations = []
for i in range(len(trajectories)):
    configuration = Configuration(atomic_numbers, trajectories[i].get_positions(), cell, pbc, None, np.array([[target[i]]]))    
    configurations.append(configuration)

dataset = create_dataset_from_configurations(configurations, z_table, cutoff=10.0, show_progress=False)
for i, data_list in enumerate(dataset["data_list"]):
    data_list["graph_labels"] = i

datamodule = DictModule(dataset, lengths=[0.8,0.2], batch_size=32)

### Create Model ###

model = SchNetModel(n_out=1, cutoff=dataset.metadata["cutoff"], atomic_numbers=dataset.metadata["atomic_numbers"], w_out_after_pool=True)
model = RegressionCV(model)

### Define Trainer ###

metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", patience=10, min_delta=1e-5)

trainer = Trainer(callbacks=[metrics, early_stopping], enable_model_summary=False)

### Optimization ###

trainer.fit(model, datamodule)

### Plots ###

ax = plot_metrics(metrics.metrics, keys=['train_loss_epoch','valid_loss'], linestyles=['-.','-'], colors=['fessa1','fessa5'], yscale='log')
plt.savefig('plot_metrics.svg')

pred = model(dataset.get_graph_inputs()).detach().squeeze()
figures.pred(target, pred)

### Compile ###

model.to_torchscript('../model.ptc', method="trace")