#!/bin/bash

# $1 is the name of the conda environment

if [ -z "$1" ]
then
    echo "Please provide a name for the conda environment"
    exit 1
fi

echo "Creating conda environment $1"
conda create -n $1 python=3.8 < y && \
echo "Activating conda environment $1" && \
conda activate dl_env && \
echo "Installing dependencies" && \
# conda install cython -c anaconda && \
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch && \
echo "Done"
```
