#!/bin/bash

# install tools 
pip install pipenv
sudo apt-get install p7zip p7zip-full p7zip-rar -y

# Setup python environment
pipenv install

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







