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

### Conda environment with ASE and MACE

````
conda create -n md_env python=3.11
conda activate md_env
conda install -c conda-forge numpy pandas matplotlib scipy notebook mdtraj mdanalysis py-plumed cmake
pip install torch torchvision cuequivariance cuequivariance-torch ase mlcolvar chemiscope rich
````

### Luigi Bonati fork of MACE

````
git clone https://github.com/luigibonati/mace.git
pip install ./mace
````

I downloaded mace-Fe111-charges.model from the link he sent me.

### Libtorch

To know which version to download visit the website : https://pytorch.org/get-started/locally/

````
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.11.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.11.0+cu126.zip
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
wget https://github.com/plumed/plumed2/releases/download/v2.10.0/plumed-2.10.0.tgz
tar -xf plumed-2.10.0.tgz
cd plumed-2.10.0
./configure --prefix=$HOME/plumed-opes-libtorch --enable-libtorch --enable-modules=opes+pytorch
make -j$(nproc)
make install
nano ~/.bashrc
````

add to the end of the file

````
# >>> plumed initialize >>>
export PATH=$HOME/plumed-opes-libtorch/bin:$PATH
export C_INCLUDE_PATH=$HOME/plumed-opes-libtorch/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$HOME/plumed-opes-libtorch/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/plumed-opes-libtorch/lib/pkgconfig:$PKG_CONFIG_PATH
export PLUMED_KERNEL=$HOME/plumed-opes-libtorch/lib/libplumedKernel.so
# <<< plumed initialize <<<
````

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
plumed config module opesP
echo $PLUMED_KERNEL
which gmx_mpi
````

## Advises

It is strongly advised to disable the autoinstalled extension "Python Environments" if your working on VS Code.