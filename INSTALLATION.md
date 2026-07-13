# Installation

This file summarize all the instruction to reproduct the same setup used during this project.

The project uses together the ASE python package, the MACE machine learning potentials and PLUMED with the OPES and LIBTORCH extenstions.

The MACE model used to get charges was made by Luigi Bonati and needed the installation of his fork of MACE as well as to download a model he pretrained for the system studied.

---

## Setup

Remove other conda environments with PLUMED and/or GROMACS.

---

## Instructions

### Miniconda

remove .conda and conda related things in .bashrc

````
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash $HOME/Miniconda3-latest-Linux-x86_64.sh
exec bash
$HOME/miniconda3/condabin/conda
conda config --set auto_activate_base false
exec bash
conda config --set channel_priority strict
conda config --add channels conda-forge
exec bash
````

comment with # `default` in .condarc and miniconda3/.condarc 

<!-- had issues with franken because of cupy so had to install cuda-toolkit -->
<!-- conda install py-plumed "numpy<2.0 ? -->
<!-- installed code for commitor with: pip install git+https://github.com/luigibonati/ase-tps.git -->

### Luigi Bonati fork of MACE (for MACECharge)

````
git clone https://github.com/luigibonati/mace.git
pip install -e mace
````

I downloaded mace-Fe111-charges.model from the link he sent me.

Installed the MACE-OFF medium24 foundation model from : https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off24/MACE-OFF24_medium.model

### Luigi Bonati fork of ASE (for ExtraCV)

````
git clone https://github.com/luigibonati/ase.git -b feat_plumed_extracv
pip install -e ase
````

### Latest version of mlcolvar (for GNN)

````
git clone https://github.com/luigibonati/mlcolvar.git -b release/2.0
pip install -e mlcolvar
````

### ORCA

Register and download the right version of ORCA on www.faccts.de.

````
tar -xf orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg.tar.xz
mkdir ~/.config/ase
vim ~/.config/ase/config.ini
````

insert by pressing I

````
[orca]
command = /home/usr/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca
````

change usr with your username
exit by pressing Escape and writing :wq!

````
nano ~/.bashrc
````

add to the end of the file

````
# >>> orca initialize >>>
export ORCA_DIR=$HOME/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
export PATH=$ORCA_DIR:$PATH
# <<< orca initialize <<<
````

change usr with your username
save and exit with Ctrl+O, Enter, Ctrl+X

````
source ~/.bashrc
````

## Advises

It is strongly advised to disable the autoinstalled extension "Python Environments" if your working on VS Code.
