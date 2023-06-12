#!/bin/bash

mkdir -p $HOME/local
mkdir -p $HOME/local/bin
mkdir -p $HOME/local/include
mkdir -p $HOME/local/lib

#add the following to your bashrc...
echo 'export PATH="$PATH:$HOME/local/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/local/lib"'  >> ~/.bashrc
echo 'export LIBRARY_PATH="$LIBRARY_PATH:$HOME/local/lib"' >> ~/.bashrc
echo 'export CPATH="$CPATH:$HOME/local/include"' >> ~/.bashrc

echo -e "\nStart zlib-1.2.13 install"
wget "https://www.zlib.net/zlib-1.2.13.tar.gz"
tar axf zlib-1.2.13.tar.gz
cd zlib-1.2.13
cd build
./configure --prefix=$HOME/local
make
make check
make install
cd ..
rm zlib-1.2.13.tar.gz

wget "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz"
tar axf hdf5-1.10.5.tar.gz
cd hdf5-1.10.5
./configure --prefix=$HOME/local --enable-cxx
make
make check
make install
cd ..
rm hdf5-1.10.5.tar.gz


wget "https://github.com/Unidata/netcdf-c/archive/v4.6.2.tar.gz"
tar axf "v4.6.2.tar.gz"
cd netcdf-c-4.6.2
./configure --prefix=$HOME/local --enable-netcdf-4
make
make check
make install
cd ..
rm v4.6.2.tar.gz

source ~/.bashrc

wget "https://github.com/Unidata/netcdf-cxx4/archive/refs/tags/v4.2.1.tar.gz"
tar axf v4.2.1.tar.gz
cd netcdf-cxx4-4.2.1/
./configure --prefix=$HOME/local
make
make check
make install
cd ..
rm v4.2.1.tar.gz

wget "https://github.com/Unidata/netcdf-cxx4/archive/refs/tags/v4.3.0.tar.gz"
tar axf v4.3.0.tar.gz
cd netcdf-cxx4-4.3.0
./configure --prefix=$HOME/local
make
make check
make install
cd ..
rm v4.3.0.tar.gz
