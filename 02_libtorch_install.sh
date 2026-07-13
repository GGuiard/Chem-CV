#!/bin/bash
# ============================================================
# 02_libtorch_install.sh
# ============================================================

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate md_env

TORCH_PREFIX=$(python -c "import torch; print(torch.__file__.rsplit('/',1)[0])")

mkdir $HOME/libtorch
cat > $HOME/libtorch/sourceme.sh << EOF
export LIBTORCH=${TORCH_PREFIX}
export CONDA_LIB=$HOME/miniconda3/envs/md_env/lib

export LD_LIBRARY_PATH=\${CONDA_LIB}:\${LIBTORCH}/lib:\$LD_LIBRARY_PATH
export CPATH=\${LIBTORCH}/include:\${LIBTORCH}/include/torch/csrc/api/include:\$CPATH
export LIBRARY_PATH=\${LIBTORCH}/lib:\$LIBRARY_PATH
export CMAKE_PREFIX_PATH=\${LIBTORCH}:\$CMAKE_PREFIX_PATH
EOF 

echo ". $PWD/sourceme.sh" >> ~/.bashrc