#!/usr/bin/env bash

sudo apt update -y
sudo apt upgrade -y
sudo apt install nvidia-cuda-toolkit -y

sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

python3 -m pip install --upgrade pip
pip3 install cython
pip3 install pycuda --user
