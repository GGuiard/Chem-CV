from mlcolvar.data import DictDataset, DictModule
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import plumed
from ase.io import read
import matplotlib.pyplot as plt

import os
os.chdir("N2-Fe")

### Import data ###

traj = read("traj_comp.traj", ":")
_, d, c, _, _, _ = plumed.read_as_pandas("COLVAR").to_numpy().T
q = np.loadtxt("CHARGES")

### Create Dataset ###

X = np.array([d, c]).T
y = (q[:,72]+q[:,73])/2

dataset = DictDataset(dict(data=Tensor(X), target=Tensor(y)))
datamodule = DictModule(dataset, lengths=[0.8,0.2], batch_size=1024)

### Create Model ###

layers = [X.shape[1],25,50,25,1]
nn_args = {'activation': 'relu'}
norm_args = {}

model = RegressionCV(layers, options={'norm_in':norm_args, 'nn':nn_args})

### Define Trainer ###

metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", patience=10, min_delta=1e-5)

trainer = Trainer(callbacks=[metrics, early_stopping])

### Optimization ###

trainer.fit(model, datamodule)

### Plots ###

ax = plot_metrics(metrics.metrics, keys=['train_loss_epoch','valid_loss'], linestyles=['-.','-'], colors=['fessa1','fessa5'], yscale='log')

plt.show()

### Compile ###

model.to_torchscript('model.ptc', method="trace")