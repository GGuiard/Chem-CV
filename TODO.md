# What is yet to do ?

This file compiles the different things to do for the advancement of the project, along propositions of modifications to try.

Overall the code need more comments, at least for the functions, with specified argument type.

---

### INSTALLATION.md

- [ ] Change export PYTHONPATH in ~/.bashrc to a more secure option

## N2-Fe

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
- [X] Setup save options, location, name, dpi, transparent...
- [X] Add color to 2D trj plots
- [ ] Make plots to show where the bias is added and what it looks like
- [ ] Make multiple plots with all the trj, all the fes
- [X] Fix min and max of colorbar in charge plot

### init_system.py

- [ ] Find a way to add depth to the cell as wanted

### charge.py

- [ ] See if it is possible to get charges for multiples atoms, with GPU parallelization
- [ ] Reduce memory surloading

### mlcharge.py

- [ ] Export model with torch.export.export
- [ ] Implement other descriptors as : LOCAL_CRISTALINITY, CONTACTMAP, KDE of DISTANCE