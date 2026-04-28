import os
os.chdir("orca_parser")

from ChemCV import ChemCV

chemcv = ChemCV(nb_traj=1)
chemcv.update()
# print(chemcv)
# print(chemcv.summary())
chemcv.save_json("CHEMCV")