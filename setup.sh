#!/bin/bash

# install tools 
pip install pipenv
sudo apt-get install p7zip p7zip-full p7zip-rar -y

# Download data
mkdir -p data-unversioned/
cd data-unversioned/
wget https://zenodo.org/record/3723295/files/annotations.csv 
wget https://zenodo.org/record/3723295/files/candidates.csv 
wget https://zenodo.org/record/3723295/files/subset0.zip
7e e data-unversioned/subset0.zip
#for i in {0..6};
#do
#    wget https://zenodo.org/record/3723295/files/subset$i.zip
#    mkdir -p subset$i
#    7e e subset$i.zip
#    rm subset$i.zip
#done
cd ../





