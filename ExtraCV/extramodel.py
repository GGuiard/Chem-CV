import numpy as np
import torch
from ase.io import read

def extracv_function(atoms):
    cv = atoms.get_positions()[1,0] - atoms.get_positions()[0,0]

    grad_cv = np.zeros((len(atoms), 3))
    grad_cv[0, 0] = -1.0
    grad_cv[1, 0] = 1.0

    return cv, grad_cv

class extracv_torch:
    def __init__(self, path):
        self.model = torch.jit.load(path)
        self.model.eval()

    def __call__(self, atoms):
        s = [atoms.get_positions()[0,0], atoms.get_positions()[1,0]]
        grad_s = np.zeros((2, len(atoms), 3))
        grad_s[0,0,0], grad_s[1,1,0] = 1.0, 1.0

        s = torch.tensor([s], dtype=torch.float32, requires_grad=True)
        cv = self.model(s)
        grad_f = torch.autograd.grad(outputs=cv, inputs=s, grad_outputs=torch.ones_like(cv))[0].numpy()[0]

        cv = cv.detach().numpy()[0]
        grad_cv = np.tensordot(grad_f, grad_s, axes=1)

        return cv, grad_cv
    
# s and grad_s should be given to extracv_torch instead of atoms
# and atoms could be one of the "CVs" with already implemented grads
# atoms should be a boolean option
# if other cv than atom should be implemented in ase plumed calc directly, not in the calculator

# class extracv_torch_ase:
#     def __init__(self, path):
#         self.model = torch.jit.load(path)
#         self.model.eval()

#     def __call__(self, s, grad_s):
#         s = torch.tensor([s], dtype=torch.float32, requires_grad=True)
#         cv = self.model(s)
#         grad_f = torch.autograd.grad(outputs=cv, inputs=s, grad_outputs=torch.ones_like(cv))[0].numpy()[0]

#         cv = cv.detach().numpy()[0]
#         grad_cv = np.tensordot(grad_f, grad_s, axes=1)

#         return cv, grad_cv
    
### Tests ###

# import os
# os.chdir("ExtraCV")

# atoms = read("init.xyz")
# extracv = extracv_torch("model.ptc")
# cv, grad_cv = extracv(atoms)
# print(cv)
# print(grad_cv)