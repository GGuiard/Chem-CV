import os
os.chdir("debug GNN")

from mlcolvar.data import DictModule
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt

### Create Dataset ###

target = [-0.23, -0.30, -0.40, -0.32, -0.28, -0.32, -0.40, -0.30, -0.23] # Mulliken charge on the central carbon

dataset = create_dataset_from_trajectories("traj_SN2.xyz", topologies=None, cutoff=10.0, graph_labels=target)

datamodule = DictModule(dataset, lengths=[0.8,0.2])

### Create Model ###

model = SchNetModel(n_out=1, cutoff=dataset.metadata["cutoff"], atomic_numbers=dataset.metadata["atomic_numbers"], w_out_after_pool=True)
model = RegressionCV(model)

### Define Trainer ###

metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", patience=10, min_delta=1e-5)

trainer = Trainer(callbacks=[metrics, early_stopping], enable_model_summary=False)

pred = model(dataset.get_graph_inputs()).detach().squeeze()
fig, ax = plt.subplots()
ax.plot([-1, 1], [-1, 1], color='k', linestyle='--')
ax.plot(target, pred, 'o', ms=1)
ax.set_xlabel("Reference")
ax.set_ylabel("Prediction")
fig.savefig("pred_beforeopt.svg")

### Optimization ###

trainer.fit(model, datamodule)

### Plots ###

ax = plot_metrics(metrics.metrics, keys=['train_loss_epoch','valid_loss'], linestyles=['-.','-'], colors=['fessa1','fessa5'], yscale='log')
plt.savefig('plot_metrics.svg')

pred = model(dataset.get_graph_inputs()).detach().squeeze()
fig, ax = plt.subplots()
ax.plot([-1, 1], [-1, 1], color='k', linestyle='--')
ax.plot(target, pred, 'o', ms=1)
ax.set_xlabel("Reference")
ax.set_ylabel("Prediction")
fig.savefig("pred_afteropt.svg")