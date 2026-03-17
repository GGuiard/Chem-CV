<!-- removed other env with plumed and gromacs from conda -->

conda create -n md_env python=3.11
conda activate md_env
conda install -c conda-forge numpy pandas matplotlib scipy notebook mdtraj mdanalysis cmake
pip install torch torchvision
pip install mace-torch
pip install ase

wget https://github.com/plumed/plumed2/releases/download/v2.10.0/plumed-2.10.0.tgz
tar -xvzf plumed-2.10.0.tgz
cd plumed-2.10.0
./configure --prefix=$HOME/plumed-opes --enable-modules=opes
make -j$(nproc)
make install
nano ~/.bashrc
<!-- add to the end of the file -->
# >>> plumed initialize >>>
export PATH=$HOME/plumed-opes/bin:$PATH
export C_INCLUDE_PATH=$HOME/plumed-opes/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$HOME/plumed-opes/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/plumed-opes/lib/pkgconfig:$PKG_CONFIG_PATH
export PLUMED_KERNEL=$HOME/plumed-opes/lib/libplumedKernel.so
# <<< plumed initialize <<<
<!-- save and exit with Ctrl+O, Enter, Ctrl+X -->
source ~/.bashrc

cd ~
wget https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz
tar -xvzf gromacs-2024.3.tar.gz
cd gromacs-2024.3
plumed patch -p
<!-- enter the number of the corresponding version of gromacs -->
mkdir build
cd build
cmake .. -DGMX_THREAD_MPI=OFF -DGMX_MPI=ON -DGMX_BUILD_OWN_FFTW=ON -DCMAKE_INSTALL_PREFIX=$HOME/gromacs-plumed
make -j$(nproc)
sudo make install
nano ~/.bashrc
<!-- add to the end of the file -->
# >>> gmx_mpi initialize >>>
source $HOME/gromacs-plumed/bin/GMXRC
# <<< gmx_mpi initialize <<<
<!-- save and exit with Ctrl+O, Enter, Ctrl+X -->
source ~/.bashrc
