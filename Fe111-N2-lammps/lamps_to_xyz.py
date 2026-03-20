import pandas as pd
import numpy as np

df = pd.read_csv("Fe111-N2-lammps/input.xyz", sep=' ')

with open("Fe111-N2-lammps/output.xyz", "w") as file:
    for i in range(len(df['S'])):
        if df['S'][i]==1:
            file.write('Fe')
        elif df['S'][i]==2:
            file.write('N ')
        file.write(f"{df['X'][i]:17.8f}{df['Y'][i]:17.8f}{df['Z'][i]:17.8f}\n")