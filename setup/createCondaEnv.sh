#!/bin/bash

# Create Environment Eventfulness
conda create -n eventfulness python=3.9
CONDA_PATH=$(which conda)
CONDA_DIR=${CONDA_PATH%/*}
CONDA_DIR=${CONDA_DIR%/*}
source "${CONDA_DIR}"/etc/profile.d/conda.sh
conda activate eventfulness

# Change this pytorch1.9.1 installation line based on your machine specification: https://pytorch.org/get-started/previous-versions/#v191
# Also, please use the pip option provided by pytorch since conda would take forever to resolve the conflicts
# The program assume that a CUDA GPU is available, so you should build this on an environment with CUDA GPU
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install librosa matplotlib
pip3 install psutil
pip3 install torchmetrics
pip3 install tensorboard
pip3 install setuptools==59.5.0
pip3 install setuptools
pip3 install protobuf
pip3 install audioread
pip3 install six
pip3 install av