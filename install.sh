#!/bin/bash
# Install conda enviroment
conda env create -f environment.yml
conda activate OneSLAM

# Install g2opy (TODO: Make sure cmake is installed)
git clone https://github.com/uoip/g2opy.git

# -------------------------------
# Apply fixes to downloaded files
cp ./misc/fixed_g2opy_files/eigen_types.h ./g2opy/python/core/eigen_types.h
cp ./misc/fixed_g2opy_files/setup.py ./g2opy/setup.py
# -------------------------------

# Continue installing
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install

# Return to root
cd ..
