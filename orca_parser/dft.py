import os
os.chdir("orca_parser")

from ChemCV import ChemCV

chemcv = ChemCV()
simpleinput, blocks = chemcv.get_orca_input()
chemcv.update()
chemcv.save()