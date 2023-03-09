#!/bin/bash
BASEDIR=$PWD

mkdir -p $BASEDIR/local
mkdir -p $BASEDIR/local/bin
mkdir -p $BASEDIR/local/include
mkdir -p $BASEDIR/local/lib

#add the following to your bashrc...
export PATH="$PATH:$BASEDIR/local/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BASEDIR/local/lib"
export LIBRARY_PATH="$LIBRARY_PATH:$BASEDIR/local/lib"
export CPATH="$CPATH:$BASEDIR/local/include"

echo -e "\nStart zlib-1.2.11 install"
wget "https://www.zlib.net/zlib-1.2.11.tar.gz"
tar axf zlib-1.2.11.tar.gz
cd zlib-1.2.11
cd build
./configure --prefix=$BASEDIR/local
make
make check
make install
cd ..

wget "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/hdf5-1.10.4.tar.gz"
tar axf hdf5-1.10.4.tar.gz
cd hdf5-1.10.4
./configure --prefix=$BASEDIR/local --enable-cxx
make
make check
make install
cd ..


wget "https://github.com/Unidata/netcdf-c/archive/v4.6.2.tar.gz"
tar axf "v4.6.2.tar.gz"
cd netcdf-c-4.6.2
./configure --prefix=$HOME/local --enable-netcdf-4
make
make check
make install
cd ..

wget "ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-cxx-4.2.tar.gz"
tar axf netcdf-cxx-4.2.tar.gz
cd netcdf-cxx-4.2
./configure --prefix=$HOME/local
make
make check
make install
cd ..


wget "https://github.com/Unidata/netcdf-cxx4/archive/v4.3.0.tar.gz"
tar axf v4.3.0.tar.gz
cd netcdf-cxx4-4.3.0
./configure --prefix=$HOME/local
make
make check
make install
cd ..
