# INSTALLATION {#install}

A general makefile is included with the repository. To install on your system, update the CUDA_LIB,
CUDA_INC, LIB_CUDA, LIB_CGAL, LIB_NETCDF paths, and make sure the PATH and LD_LIBRARY_PATH
environment variables are appropriately set. From there a simple "make" will do the trick. See below
for detailed installation instructions on MacOS. The command "make float" will compile the code with
everything in floating-point precision (a bit faster on GPUs, but, of course, less numerically precise).
The command "make debug" will add common debugging flags, and also enforce always-reproducible random
number generation.

# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using
CUDA-8.0. The code has been tested with CUDA versions as early as 6.5, and uses compute capability
3.5 devices and higher. It ought to work on lower compute capability devices; compile without the
"-arch=sm_35" flag to run on them.

The Voronoi model branch uses the CGAL library; this dependency can be removed, if necessary, by monkeying
with the code to run "fullTriangulation()" rather than "globalTriangulationCGAL()" in the relevant
spots. This is highly discouraged, and the code may be much less stable as a result. In any event,
CGAL-4.9 was used, which in turn requires up-to-date versions of the gmp and mpfr libraries.
The code was developed and tested against gmp-6.1.2 and mpfr-3.1.5.

The default database class uses the netCDF-4 C++  library (tested on version 4.1.3).The dependency on netCDF
can be removed by (1) not including any "Database" class, and (2) commenting out everything after the
= sign in the LIB_NETCDF entry of the makefile

The calculation of the dynamical matrix makes use of Eigen3.3.3

Documentation is maintained via doxygen, but is not required for compilation of the executables.

# Sample programs

This repository comes with sample main cpp files that can be compiled into executables in both the root directory
and in examples/. Please see the [examples](@ref code) documentation for details on each.
range of parameters.

# Mac OS X Instructions

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
