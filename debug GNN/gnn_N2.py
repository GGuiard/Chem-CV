import os
os.chdir("debug GNN")

from mlcolvar.data import DictModule
from mlcolvar.data.utils import save_dataset_configurations_as_extyz
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from mlcolvar.utils.io import create_dataset_from_configurations, create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.data.graph.atomic import Configuration, AtomicNumberTable

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

import torch
# torch.set_default_dtype(torch.float64)

### Create Dataset ###

target = [0.57422   , 0.610327  , 0.609769  , 0.5740615 , 0.5439325 , 0.5691155 , 0.610928,
          0.618973  , 0.5857015 , 0.5446485 , 0.557408  , 0.6010635 , 0.6142095 , 0.5861845,
          0.5445215 , 0.555298  , 0.6019175 , 0.6200305 , 0.595599  , 0.546754  , 0.5426955,
          0.5881395 , 0.611442  , 0.5900675 , 0.5372765 , 0.524902  , 0.568295  , 0.5943605,
          0.5713795 , 0.5211335 , 0.52178   , 0.5758025 , 0.6010665 , 0.570432  , 0.5180675,
          0.5304595 , 0.587551  , 0.6060345 , 0.5697075 , 0.5153585 , 0.5257575 , 0.5754585,
          0.58636   , 0.5491495 , 0.5091675 , 0.5358495 , 0.5844755 , 0.594999  , 0.563241,
          0.5290365 , 0.5520795 , 0.6027245 , 0.6216555 , 0.598386  , 0.554243  , 0.5478435,
          0.587174  , 0.615976  , 0.605565  , 0.5586955 , 0.5224195 , 0.552261  , 0.6032575,
          0.6117895 , 0.570671  , 0.5298775 , 0.5601665 , 0.609277  , 0.6155455 , 0.5768595,
          0.5373485 , 0.564627  , 0.6114365 , 0.6135595 , 0.569943  , 0.541366  , 0.5824675,
          0.635556  , 0.647905  , 0.617712  , 0.581113  , 0.5991945 , 0.6435555 , 0.65315,
          0.617566  , 0.5586855 , 0.5409285 , 0.5700285 , 0.5833555 , 0.550804  , 0.495979,
          0.495367  , 0.5483315 , 0.571185  , 0.5382135 , 0.496143  , 0.5241565 , 0.5792755,
          0.5887655 , 0.546811 ] # Mean charge on N2

dataset = create_dataset_from_trajectories("traj_N2.xyz", topologies=None, cutoff=4.0, graph_labels=target, system_selection="type N", environment_selection="not type N")
save_dataset_configurations_as_extyz(dataset, "test.xyz")
datamodule = DictModule(dataset, lengths=[0.8,0.2])

# trajectories = read("traj_N2.traj", ":")

# atomic_numbers = trajectories[0].get_atomic_numbers()
# cell = np.array(trajectories[0].get_cell())
# pbc = trajectories[0].get_pbc()
# z_table = AtomicNumberTable.from_zs(atomic_numbers)

# configurations = []
# for i in range(len(trajectories)):
#     configuration = Configuration(atomic_numbers, trajectories[i].get_positions(), cell, pbc, None, np.array([[target[i]]]))    
#     configurations.append(configuration)

# dataset = create_dataset_from_configurations(configurations, z_table, cutoff=10.0, show_progress=False)
# for i, data_list in enumerate(dataset["data_list"]):
#     data_list["graph_labels"] = i

# datamodule = DictModule(dataset, lengths=[0.8,0.2], batch_size=32)

### Create Model ###

gnn_model = SchNetModel(n_out=1, cutoff=dataset.metadata["cutoff"], atomic_numbers=dataset.metadata["atomic_numbers"], w_out_after_pool=False, n_layers=2)

options = {'optimizer': {'lr': 1e-3}}
model = RegressionCV(gnn_model, options=options)

### Define Trainer ###

metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", patience=5000, min_delta=1e-5)

trainer = Trainer(callbacks=[metrics, early_stopping], enable_model_summary=False, max_epochs=5000)

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