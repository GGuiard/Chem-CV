import os
os.chdir("ExtraCV")

import numpy as np
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.cvs import RegressionCV
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from torch import Tensor
from torch import randn
from torch.jit import trace
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import plumed
from ase.io import read
import matplotlib.pyplot as plt

### Import data ###

_, x1, x2, d = plumed.read_as_pandas("COLVAR").to_numpy().T

### Create Dataset ###

X = np.array([x1, x2]).T
y = np.array([d]).T

dataset = DictDataset(dict(data=Tensor(X), target=Tensor(y)))
datamodule = DictModule(dataset, lengths=[0.8,0.2], batch_size=1024)

### Create Model ###

layers = [X.shape[1], 1]
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

trace(model, randn(1, 2)).save("model.ptc")