import numpy as np

def extracv(atoms):
    cv = atoms.get_positions()[1,0] - atoms.get_positions()[0,0]

    grad_cv = np.zeros((len(atoms), 3))
    grad_cv[0, 0] = -1.0
    grad_cv[1, 0] = 1.0

    return cv, grad_cv