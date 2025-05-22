#!/bin/bash

mkdir -p $HOME/.local
mkdir -p $HOME/.local/bin
mkdir -p $HOME/.local/include
mkdir -p $HOME/.local/lib
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.local/lib"'  >> ~/.bashrc
echo 'export LIBRARY_PATH="$LIBRARY_PATH:$HOME/.local/lib"' >> ~/.bashrc
echo 'export CPATH="$CPATH:$HOME/.local/include"' >> ~/.bashrc
echo 'export GMP_LIBRARIES=$HOME/.local/lib'  >> ~/.bashrc
echo 'export GMP_INCLUDE_DIR=$HOME/.local/include' >> ~/.bashrc
echo 'export MPFR_LIBRARIES=$HOME/.local/lib' >> ~/.bashrc
echo 'export MPFR_INCLUDE_DIR=$HOME/.local/include'  >> ~/.bashrc

# install boost
wget "https://boostorg.jfrog.io/artifactory/main/release/1.84.0/source/boost_1_84_0.tar.gz"
tar -xvzf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh --prefix=$HOME/.local
./b2 install
cd ..
rm boost_1_84_0.tar.gz

# install GMP
wget "https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz"
tar -xvf gmp-6.2.1.tar.xz
cd gmp-6.2.1
./configure --prefix=$HOME/.local
make
make check
make install
cd ..
rm gmp-6.2.1.tar.xz

# install MPFR
wget "https://www.mpfr.org/mpfr-4.20/mpfr-4.2.0.tar.xz"
tar -xvf mpfr-4.2.0.tar.xz
cd mpfr-4.2.0
./configure --prefix=$HOME/.local
make
make check
make install
cd ..
rm mpfr-4.2.0.tar.xz

# install CGAL
wget "https://github.com/CGAL/cgal/releases/download/v5.6.1/CGAL-5.6.1.tar.xz"
tar -xvf CGAL-5.6.1.tar.xz
cd CGAL-5.6.1
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_BUILD_TYPE=Release .
make
make check
make install
cd ..
rm CGAL-5.6.1.tar.xz

# install zlib
wget "https://www.zlib.net/fossils/zlib-1.2.13.tar.gz"
tar axf zlib-1.2.13.tar.gz
cd zlib-1.2.13
./configure --prefix=$HOME/.local
make
make check
make install
cd ..
rm zlib-1.2.13.tar.gz

# install hdf5
wget "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz"
tar axf hdf5-1.10.5.tar.gz
cd hdf5-1.10.5
./configure --prefix=$HOME/.local --enable-cxx
make
make check
make install
cd ..
rm hdf5-1.10.5.tar.gz

# install netcdf-c
wget "https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/netcdf-c-4.9.2.tar.gz"
tar axf "netcdf-c-4.9.2.tar.gz"
cd netcdf-c-4.9.2
./configure --prefix=$HOME/.local --enable-netcdf-4 --disable-libxml2
make
make check
make install
cd ..
rm netcdf-c-4.9.2.tar.gz

#netcdf-cxx needs to know about netcdf-c
source ~/.bashrc

# install netcdf-cxx
wget "https://downloads.unidata.ucar.edu/netcdf-cxx/4.2/netcdf-cxx-4.2.tar.gz"
tar axf netcdf-cxx-4.2.tar.gz
cd netcdf-cxx-4.2
./configure --prefix=$HOME/.local
make
make check
make install
cd ..
rm netcdf-cxx-4.2.tar.gz

# install netcdf-cxx4
wget "https://downloads.unidata.ucar.edu/netcdf-cxx/4.3.1/netcdf-cxx4-4.3.1.tar.gz"
tar axf netcdf-cxx4-4.3.1.tar.gz
cd netcdf-cxx4-4.3.1
./configure --prefix=$HOME/.local
make
make check
make install
cd ..
rm netcdf-cxx4-4.3.1.tar.gz

#install eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar axf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
cd ..
cd ..
rm eigen-3.4.0.tar.gz
