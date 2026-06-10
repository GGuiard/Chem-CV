# What is yet to do ?

This file compiles the different things to do for the advancement of the project, along propositions of modifications to try.

Overall the code need more comments, at least for the functions, with specified argument type.

## Overall

- [ ] More comments
- [ ] Docstrings for file, functions, objects and the arguments type
- [ ] Python file for general functions and notebook for system specific things
- [ ] Change overall architecture to split tasks into subfolders
- [ ] Take care of path management
- [ ] Save every self sufficient data in the subfolder associated to one's task
- [ ] Clean the repository, make an official version
- [ ] Save my_ase, my_mlcolvar, ExtraCV, ChemCV, MD-utils to other github repositories

- [ ] Export model with torch.export.export
- [ ] Take care of issues of saving and use with simple .xyz or advanced .extyz with cell and pbc for GNNs and be careful with removing the constrains
- [ ] Flush data to prevent crashing issues and memory surloading

## Markdowns

### INSTALLATION.md

- [ ] Use sourceme instead of bashrc
- [ ] Use conda torch instead of libtorch for PLUMED
- [ ] Make bash installation instructions

### README.md

- [ ] Add context
- [ ] Add a navigation
- [ ] Add figures
- [ ] Show a schematic of the pipeline

## utils

- [ ] Rename to MD-utils
- [ ] Import function from submodules instead of the whole submodule
- [ ] Make installable with pip

### md_runner.py

- [ ] Implement easy restart, autodetected, with backups

### postprocessing.py

- [X] Restrict the fes to the sampled points (density>0.01)
- [X] Set an fps option for chemiscope
- [X] Set a trajectory stride option or by default infer it from the relative sizes of COLVAR and trajectory
- [X] Set an "input" option for the transient where it is asked after showing the relevant trajectories
- [ ] The transient can be passed in percentage (float) or in a frame number (int)
- [X] Set start and end options for chemiscope

### analyze.py

- [ ] Compute the population and density using KDE instead of gaussian filter
- [ ] Compute the FES from STATES
- [X] Compute boostrap error without smoothing and then smooth it
- [ ] Generalize functions for any dimension
- [X] Change sigma to a bandwidth independant of the grid

### figures.py

- [X] Set a maximum number of points to be plotted for scatter plots
- [X] Make a plot of the log of the bias along the CV used for biasing and superimpose the FES
- [X] Make a plot showing the distribution of points with energy and their associated log bias
- [X] Make chemiscope more versatile passing structures and a dictionary, certain entries can be specified for x, y, coloring
- [ ] Add to chemiscope options to color atoms, shade the bonds, add gradient arrows...
- [X] Add an fps option for chemiscope and change its default value to 20
- [ ] Change nb_levels in grid to a level_step with 0.
- [ ] Make multiple plots with all the trj, all the fes

### helpers.py

- [ ] Implement a function to sample a trajectory according to the value of the FES or using the kernels in states

## orca_parser

- [ ] Rename it for ChemCV
- [ ] Import function from submodules instead of the whole submodule
- [ ] Make installable with pip
- [ ] Be compatible for different specified parser (even if ORCA is the only one available)
- [ ] The parser should extract the descriptors which are represented in a tree way, while ChemCV make a few combinations of them
- [ ] Implement a module for basic modeling

### orca_parser.py

- [ ] Add LED, NAO, NBO, ETS-NOCV, dot product HOMO/LUMO, groups and MO specific bond order
- [ ] Change regex to make it more readable

## my_ase

- [X] Changed the plumed calculator to convert the potential energy to float but the change should be made in franken

## my_mlcolvar

- [ ] Implement a premade plot which for a dataset or datamodule and a model show the reference over prediction graph and giving the MAE and RMSE (separating the training, validation and test datasets if presents)

### plumed_interfaces

- [X] Changed PytorchModelGNN.cpp to always implement edge_masks_lr but the change it should be optional
- [ ] Check the unit compatibility between the topology, the ase atoms, the plumed instructions, the model and the plumed interface

### mlcolvar.core.nn.graph.gnn.py

- [X] Added custom pooling for antisymmetric graph label prediction
- [ ] Add a way to specify atomic coefficients for the pooling
- [ ] Implement compatibility with a NN converting atomic nodes values to a graph label for regression tasks
- [X] Added a selection pooling returning the node values of the selected system only

### mlcolvar.data.graph.utils.py

- [X] Changed the default value of the weights in a graph dataset to a tensor
- [ ] Remove system_masks, subsystem_masks, edge_masks_lr from the default dataset construction
- [ ] Check compatibility between dataset, datamodule, configurations, graph tracing example and the plumed interface for GNN in term of weights, environment, system, subsystem, edge_maks_lr...
- [ ] Change the datamodule implementation or add a quick function to get the training, validation and test inputs and targets more easily

### mlcolvar.utils.fes.py

- [ ] Correct numerical approximation error when taking the kde with very small weights line 228
- [ ] Add the option to reduce the FES to the region explored, with a relative kde population above 0.01
- [ ] Implement bootstrap computation of the error

### mlcolvar.utils.plot.py

- [ ] Correct issues with already imported cmaps

### mlcolvar.utils.io.graphs.ase_.py

- [X] Corrected a typo in the comment line 265 : indeces -> indices

### mlcolvar.utils.io.graphs.common.py

- [X] Corrected a typo in the create_dataset_from_trajectories docstring line 109: system_selection appeared twice and was replaced on second occurence by subsystem_selection
- [X] Corrected a typo in the issue raised lin 152: wwhen -> when