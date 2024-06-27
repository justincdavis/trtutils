#!/usr/bin/env bash

# sudo apt update -y
# sudo apt upgrade -y

echo 'export PATH=${PATH}:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export CUDA_INC_DIR=/usr/local/cuda/include' >> ~/.bashrc
echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

python3 -m pip install --upgrade pip
pip3 install cython
pip3 install pycuda --user
