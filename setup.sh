#!/bin/bash

# Download data
mkdir -p data-unversioned/
cd data-unversioned/

wget https://zenodo.org/record/3723295/files/annotations.csv
wget https://zenodo.org/record/3723295/files/candidates.csv
for i in {0..6};
do
    wget https://zenodo.org/record/3723295/files/subset$i.zip
done

cd ../



