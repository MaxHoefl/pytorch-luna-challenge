#!/bin/bash

gpu_cpu=$1

# install tools 
pip install pipenv
sudo apt-get install p7zip p7zip-full p7zip-rar -y

# Setup python environment
pip uninstall -y pytest
pipenv install -d
if [ gpu_cpu = 'gpu' ];
then
    pipenv run pip install \
        torch==1.7.1+cu110 \
        torchvision==0.8.2+cu110 \
        torchaudio===0.7.2 \
        -f https://download.pytorch.org/whl/torch_stable.html
else
    pipenv run pip install \
        torch==1.7.1+cpu \
        torchvision==0.8.2+cpu \
        torchaudio==0.7.2 \
        -f https://download.pytorch.org/whl/torch_stable.html
fi

# Download data
mkdir -p data-unversioned/
cd data-unversioned/
wget https://zenodo.org/record/3723295/files/annotations.csv 
wget https://zenodo.org/record/3723295/files/candidates.csv 
wget https://zenodo.org/record/3723295/files/subset0.zip
mkdir -p subset0
7z e subset0.zip -osubset0
#for i in {0..6};
#do
#    wget https://zenodo.org/record/3723295/files/subset$i.zip
#    mkdir -p subset$i
#    7z e subset$i.zip -osubset$i
#done
cd ../







