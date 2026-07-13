#!/bin/bash
# ============================================================
# 01_conda_install.sh
# ============================================================

source $HOME/miniconda3/etc/profile.d/conda.sh

# --- md_env ---

# --- Create environment ---
conda create -y -n md_env python=3.11 pip
conda activate md_env

# --- Libraries ---
conda install -y numpy pandas matplotlib scipy notebook cython pybind11 gcc docstring_parser py-plumed
pip install torch torchvision omegaconf cuequivariance cuequivariance-torch mace-torch cupy chemiscope rich

# --- Local packages ---
pip install -e ./my_ase/
pip install -e ./my_mlcolvar/
pip install $HOME/ase-tps-main/
pip install $HOME/franken[cuda,mace] --no-deps

# --- deal ---

# --- Create environment ---
conda create -y -n deal python=3.12 pip
conda activate deal

# --- Libraries ---
conda install -y gcc gxx cmake openmp liblapacke openblas
pip install git+https://github.com/mir-group/flare.git@1.3.3b
pip install git+https://github.com/luigibonati/DEAL.git