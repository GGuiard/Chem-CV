# What is yet to do ?

This file compiles the different things to do for the advancement of the project, along propositions of modifications to try.

Overall the code need more comments, at least for the functions, with specified argument type.

---

### INSTALLATION.md

- [ ] Change export PYTHONPATH in ~/.bashrc to a more secure option
- [ ] Make different python file reusable for different systems so that to not have to copy them and update them

### main.py

- [X] Make the OPES_METAD_EXPLORE works
- [X] Save E and T somewhere accessible for the postprocessing script

### postprocessing.py

- [X] Give the possibility to save the figures into another named folder with premade names
- [X] Either wrap positions in traj or change something in chemiscope

### analyze.py

- [X] Better implement kde pop while keeping the possibility to use normal pop
- [X] Correct cum_av with weight
- [ ] Make functions understand that they need to use a parameter only if its given (ex: weights, masks, 2D)
- [ ] For bootstrap and block add the possibility to choose specify a function to apply to the data
- [ ] Generalize bootstrap block and pop to N dim

### figures.py

- [~] Adapt radius in chemiscope plots using CR from ASE (reference in DEAL repo of LB)
- [ ] Make plot with sampling superimposed on fes 2d (maybe with charge)
- [ ] Visualize STATES file and FES from STATES
- [X] Setup save options, location, name, dpi, transparent...
- [X] Add color to 2D trj plots
- [ ] Make plots to show where the bias is added and what it looks like
- [ ] Make multiple plots with all the trj, all the fes
- [X] Fix min and max of colorbar in charge plot
- [ ] Add possibility to change bond color in chemiscope

### init_system.py

- [ ] Find a way to add depth to the cell as wanted

### charge.py

- [ ] See if it is possible to get charges for multiples atoms, with GPU parallelization
- [ ] Reduce memory surloading

### mlcharge.py

- [ ] Export model with torch.export.export
- [ ] Implement other descriptors as : LOCAL_CRISTALINITY, CONTACTMAP, KDE of DISTANCE
- [X] Predict the charge using a GNN
- [ ] Find a better way to sample a trajectory (using STATES ?)

### create_traj.py

- [X] Use NEB from reactant to products
- [ ] Add a little bit of noise

### dft.py

- [X] Make a change so that each ORCA output can be saved in different directories

### orca_parser

- [X] Add options to group together orbitals of same n and l or even of same n
- [ ] Add possibility to choose options for chemcv (atoms, bonds, ao, mo...) globally and/or for each chemcv
- [X] Add threshold when relevant (p_AtMO), (like post-processing removal of zeros chemcvs)
- [ ] Add LED, NAO, NBO, ETS-NOCV, dot product HOMO/LUMO
- [ ] Change regex to make it more readable
- [ ] Change ChemCV so that it can be changed or copied in a way that adapt the active_chemcvs, their options, and the treeframe 
- [X] Regroup save and load for json and hdf5 with an auto-mode
- [X] Save and load active_chemcv
- [X] Split into two objects: a multi-index dataframe with all its changes and the chemcv
- [X] Change update so that a directory can be specified
- [X] Move parsing functions to another file (as well as the available chemcv dictionary ?)
- [X] Make the dataframe initialize on the first update (and on other updates if new indexes are discovered it should either create new columns and complete them with 0 or raise a warning)
- [X] Make parsing functions return (MD) dictionnary
- [X] Make a greedy or tolerant option which respectively decides if an absent index should mean the deletion of a column and if a new index should mean the creation of a column