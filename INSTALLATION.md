# Installation

This file summarize all the instruction to reproduct the same setup used during this project.

The project uses together the ASE python package, the MACE machine learning potentials and PLUMED with the OPES and LIBTORCH extenstions.

The MACE model used to get charges was made by Luigi Bonati and needed the installation of his fork of MACE as well as to download a model he pretrained for the system studied.

It also show how to install GROMACS and patch it with the version of PLUMED compiled, which is not needed for this project but was useful elsewhere.

---

## Setup

Remove other conda environments with PLUMED and/or GROMACS.

---

## Instructions

### Conda environment

````
conda create -n md_env python=3.11
conda activate md_env
conda install -c conda-forge numpy pandas matplotlib scipy notebook cython pybind11 pytables
pip install torch torchvision cuequivariance cuequivariance-torch chemiscope rich franken[cuda,mace]
````

if needed install cmake with conda

<!-- had issues with franken because of cupy so had to install cuda-toolkit -->
<!-- conda install py-plumed "numpy<2.0 ? -->
<!-- installed code for commitor with: pip install git+https://github.com/luigibonati/ase-tps.git -->

<!-- On CINECA:
conda create -n md_env python=3.11 pip libstdcxx-ng numconda py pandas matplotlib scipy
conda activate md_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cuequivariance cuequivariance-torch ase mace-torch

pip install ./mlcolvar/ ./ase-tps-main/
 
module load cuda/12.2
module load nvhpc/
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDAARCHS=80
pip install cupy-cuda12x
git clone https://github.com/CSML-IIT-UCL/franken.git
pip install ./franken[cuda,mace] --no-deps
-->

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

### Libtorch

To know which version to download visit the website : https://pytorch.org/get-started/locally/

````
wget https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip
unzip libtorch-shared-with-deps-2.11.0+cu130.zip
````

Save the path in .bashrc :

````
nano ~/.bashrc
````

Add to the end of the file :

````
# >>> libtorch initialize >>>
export CPATH=$HOME/libtorch/include/torch/csrc/api/include/:$HOME/libtorch/include/:$HOME/libtorch/include/torch:$CPATH
export INCLUDE=$HOME/libtorch/include/torch/csrc/api/include/:$HOME/libtorch/include/:$HOME/libtorch/include/torch:$INCLUDE
export LIBRARY_PATH=$HOME/libtorch/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH
# >>> libtorch initialize >>>
````

Save and exit with Ctrl+O, Enter, Ctrl+X.

````
source ~/.bashrc
````

### PLUMED with OPES and LIBTORCH

````
conda activate md_env
wget https://github.com/plumed/plumed2/releases/download/v2.10.0/plumed-2.10.0.tgz
tar -xf plumed-2.10.0.tgz
cd plumed-2.10.0
./configure --prefix=$HOME/plumed-opes-libtorch-python PYTHON_BIN=$(which python) PYTHON_CONFIG=$(which python3-config) --enable-python --enable-libtorch --enable-modules=opes+pytorch --enable-modules=opes+pytorch
make -j$(nproc)
make install
nano ~/.bashrc
````

add to the end of the file

````
# >>> plumed initialize >>>
export PATH=$HOME/plumed-opes-libtorch-python/bin:$PATH
export C_INCLUDE_PATH=$HOME/plumed-opes-libtorch-python/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$HOME/plumed-opes-libtorch-python/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/plumed-opes-libtorch-python/lib/pkgconfig:$PKG_CONFIG_PATH
export PLUMED_KERNEL=$HOME/plumed-opes-libtorch-python/lib/libplumedKernel.so
export PYTHONPATH=$HOME/plumed-opes-libtorch-python/lib/plumed/python:$PYTHONPATH
# <<< plumed initialize <<<
````

save and exit with Ctrl+O, Enter, Ctrl+X

````
source ~/.bashrc
````

Might need to specify the plumed source to load plumed interfaces from mlcolvar:

````
PLUMED_SOURCE=$HOME/plumed-2.10.0/sourceme.sh
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

### GROMACS patched with PLUMED

````
cd ~
conda activate md_env
wget https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz
tar -xf gromacs-2024.3.tar.gz
cd gromacs-2024.3
plumed patch -p
````

enter the number of the corresponding version of gromacs

````
mkdir build
cd build
cmake .. -DGMX_THREAD_MPI=OFF -DGMX_MPI=ON -DGMX_BUILD_OWN_FFTW=ON -DCMAKE_INSTALL_PREFIX=$HOME/gromacs-plumed
make -j$(nproc)
sudo make install
nano ~/.bashrc
````

add to the end of the file

````
# >>> gmx_mpi initialize >>>
source $HOME/gromacs-plumed/bin/GMXRC
# <<< gmx_mpi initialize <<<
````

save and exit with Ctrl+O, Enter, Ctrl+X

````
source ~/.bashrc
````

---

## Checks

check installation after restarting terminal

````
which plumed
plumed config module opes
echo $PLUMED_KERNEL
which gmx_mpi
````

## Advises

It is strongly advised to disable the autoinstalled extension "Python Environments" if your working on VS Code.
