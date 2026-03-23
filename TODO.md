# What is yet to do ?

This file compiles the different things to do for the advancement of the project, along propositions of modifications to try.

Overall the code need more comments, at least for the functions, with specified argument type.

---

## N2-Fe

### main.py

- [ ] Make the OPES_METAD_EXPLORE works
- [ ] Save E and T somewhere accessible for the postprocessing script
- [ ] Give the possibility to save the COLVAR and traj fills into another named folder

### postprocessing.py

- [ ] Give the possibility to save the figures into another named folder

### analyze.py

- [ ] Better implement kde pop while keeping the possibility to use normal pop
- [ ] Make functions understand that they need to use a parameter only if its given (ex: weights, masks, 2D)
- [ ] For bootstrap and block add the possibility to choose specify a function to apply to the data
- [ ] Initialize the scrip with the temperature
- [ ] Generalize bootstrap block and pop to N dim

### figures.py

- [ ] Add the population with its error to 1D fes plots
- [ ] Setup save options, location, name, dpi, transparent...
- [ ] Make animations or add color to 2D trj plots
- [ ] Make plots to show where the bias is added and what it looks like
- [ ] Make multiple plots with all the trj, all the fes
- [ ] Adapt the plots to the system studied by adding a label input
- [ ] Set aspect ratio 1:1 when relevant
- [ ] Give the possibility to not show err in 1D fes plots