#!/bin/bash
# ============================================================
# 03_plumed_install.sh
# ============================================================

# --- Source libtorch first ---
source $HOME/libtorch/sourceme.sh

# --- Activate conda (for python bindings) ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate md_env

# --- Download and extract PLUMED ---
cd $HOME
wget https://github.com/plumed/plumed2/releases/download/v2.10.0/plumed-2.10.0.tgz
tar -xf plumed-2.10.0.tgz
cd $HOME/plumed-2.10.0

# --- Configure ---
mpicompiler=$(which mpicxx)
PLUMED_INSTALL=$HOME/plumed-opes-libtorch-python

./configure \
    --prefix=${PLUMED_INSTALL} \
    CXX=${mpicompiler} \
    CXXFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include -I${LIBTORCH}/include -D_GLIBCXX_USE_CXX11_ABI=1" \
    LDFLAGS="-L${LIBTORCH}/lib -Wl,-rpath,${LIBTORCH}/lib" \
    LIBS="-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpthread -lm -ldl" \
    --enable-python \
    PYTHON_BIN=$(which python) \
    PYTHON_CONFIG=$(which python3-config) \
    --enable-libtorch \
    --enable-modules=opes+pytorch

# --- Check libtorch was detected ---
grep -q "HAS_LIBTORCH" config.log
if [ $? -ne 0 ]; then
    echo "FAILED: libtorch not detected by configure. Check config.log:"
    grep -A5 "libtorch_cuda" config.log | tail -20
    exit 1
fi
grep "HAS_LIBTORCH" config.log | tail -3

# --- Build and install ---
make -j8
make install

echo ""
echo "=== PLUMED install complete ==="

# --- Write sourceme.sh ---
cat > $HOME/plumed-opes-libtorch-python/sourceme.sh << EOF
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH
export PLUMED_KERNEL=$HOME/plumed-opes-libtorch-python/lib/libplumedKernel.so
export PLUMED_SOURCE=$HOME/plumed-2.10.0/sourceme.sh
export PATH=$HOME/plumed-opes-libtorch-python/bin:$PATH
export C_INCLUDE_PATH=$HOME/plumed-opes-libtorch-python/include:$C_INCLUDE_PATH
export PKG_CONFIG_PATH=$HOME/plumed-opes-libtorch-python/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/plumed-opes-libtorch-python/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/plumed-opes-libtorch-python/lib/plumed/python:$PYTHONPATH
EOF

echo ". $PWD/sourceme.sh" >> ~/.bashrc