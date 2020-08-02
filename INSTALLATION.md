# INSTALLATION {#install}

Version 1.0 has switched to a cmake-based build system. Most dependencies can be apt-get installed, which is convenient, and
a script for installing netcdf locally is included. As noted below, you should have the include and lib of netcdf on your PATH and LD_LIBRARY_PATH,
but as long as that's the case compilation should be straightforward. Just

* $ cd build/
* $ cmake ..
* $ make

By default this will compile executables related to the voronoi.cpp and Vertex.cpp files in the root directory.
To compile a different cpp file, got to the "foreach(ARG " line at the end of the CMakeLists.txt file and add the name of your file, without the .cpp ending, to the list.

The command "make debug" will add common debugging flags, and also enforce always-reproducible random
number generation.

# Requirements

Note that, thanks to the CGAL dependency (which, with some work, can be removed), your compiler needs to support
features of C++14. This package has been tested with gcc-6 and gcc-7.5 (and equivalent g++ versions)

The current iteration of the code was written using some features of C++14, and was compiled using
CUDA-11.0. The code has been tested with CUDA versions as early as 6.5, and uses compute capability
3.5 devices and higher.

In any event,
CGAL-5.0.2 was used, which in turn requires up-to-date versions of the gmp and mpfr libraries.
The code was developed and tested against gmp-6.1.2 and mpfr-3.1.5.All of these, including CGAL now, can be conveniently installed via apt-get

The default database class uses the netCDF-4 C++  library (tested on version 4.1.3).The dependency on netCDF
can be removed by (1) not including any "Database" class, and (2) removing the database directory and library from the cmake file

The calculation of the dynamical matrix makes use of Eigen3.3.3

Documentation is maintained via doxygen, but is not required for compilation of the executables.

# Sample programs

This repository comes with sample main cpp files that can be compiled into executables in both the root directory
and in examples/. Please see the [examples](@ref code) documentation for details on each.
range of parameters.

#Ubuntu installation

Most requirements can be obtained by the usual apt-get method; netcdf is more finicky.
An install script is included in the cmakeHelp/ directory. Modify the directory paths to be more appropriate for your system,
and use this if you prefer not to go through by hand the steps outlined on the netcdf page below. In
any event, make sure that the includes and libraries for netcdf and netcdfcxx
are on your PATH and LD_LIBRARY_PATH, respectively

# Windows Subsystem for Linux 2

This code has been tested on WSL 2, running Ubuntu 18.04, gcc-6, g++-6, and using the CUDA 11.0 toolkit

# Mac OS X Instructions

IMPORTANT NOTE: Current versions of Mac OsX do not support CUDA installation... it may be possible to hack some solution together, but for the moment it looks like Mac will not be supported. As such, these mac install instructions are for historical reference for systems running OsX 12 and earlier.

The following instructions for compiling cellGPU on Mac OSX  were contributed by Gonca Erdemci-Tandogan (https://goncaerdemci.wordpress.com/).
It does not include how to get Eigen (for the dynamical matrix diagonalization), but that is a header-only library.

Everything is most conveniently done using homebrew, the installation of which can be followed from https://brew.sh

## CGAL

brew install cgal

## netCDF

brew install homebrew/science/netcdf --with-cxx-compat

(note that you may not need the "--with-cxx-compat" flag, depending on your OS version, compiler tool chain, etc.)

## CUDA

Note: homebrew-cask drivers makes this easy, but if it doesn't work you can always download CUDA from https://developer.nvidia.com/cuda-downloads

1. ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null ; brew install caskroom/cask/brew-cask 2> /dev/null

2. brew tap caskroom/drivers

2. brew cask install cuda

3. After cuda is installed, run "which nvcc". If this returns empty, you must export the path... 

3. vim $HOME/.bashrc

3. Add the following lines to the .bashrc and .bash_profile files:
    * export PATH=/Developer/NVIDIA/CUDA-8.0/bin${PATH:+:${PATH}}
    * export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib\${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}

## Makefile changes

Following the above instructions, change the provided makefile in the following way:

1. “CUDA_LIB = /usr/local/cuda/lib64” to “CUDA_LIB = /usr/local/cuda/lib”
2. Part of “LIB_CGAL += -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr” to wherever my CGAL directory is.

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads

CGAL: http://www.cgal.org/download.html

netCDF: http://www.unidata.ucar.edu/downloads/netcdf/index.jsp (be sure to get the C++ release, not the C release)

Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page

doxygen: http://www.stack.nl/~dimitri/doxygen/download.html
