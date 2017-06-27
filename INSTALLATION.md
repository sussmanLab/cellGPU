# INSTALLATION {#install}

A general makefile is included with the repository. To install on your system, update the CUDA_LIB,
CUDA_INC, LIB_CUDA, LIB_CGAL, LIB_NETCDF paths, and make sure the PATH and LD_LIBRARY_PATH
environment variables are appropriately set.
Create the /obj and /obj/cuobj directories, and from there, a simple "make" will do the trick. See below for detailed installation instructions on MacOS
.
# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using
CUDA-8.0. The code has been tested with CUDA versions as early as 6.5, and uses compute capability
3.5 devices and higher. It ought to work on lower compute capability devices; compile without the
"-arch=sm_35" flag to run on them.

The SPV branch uses the CGAL library; this dependency can be removed, if necessary, by monkeying
with the code to run "fullTriangulation()" rather than "globalTriangulationCGAL()" in the relevant
spots. This is highly discouraged, and the code may be much less stable as a result. In any event,
CGAL-4.9 was used, which in turn requires up to date versions of the gmp and mpfr libraries.
The code was developed and tested against gmp-6.1.2 and mpfr-3.1.5.

The database class uses the netCDF-4 C++  library (tested on version 4.1.3).The dependency on netCDF
can be removed by (1) not including any "Database" class, and (2) commenting out everything after the
= sign in the LIB_NETCDF entry of the makefile

The calculation of the dynamical matrix makes use of Eigen3.3.3

Documentation is maintained via doxygen, but is not required for compilation of the executables.

The makefile system is scheduled to be replaced, and will soon change to require cmake.

# Sample programs

The included makefile compiles four programs: avmGPU, spvGPU, spvMSD, and Minimize.out. The first two generate timing
information, and the "Timing.sh" script shows an example of how to quickly call these programs for a
range of parameters.

The third program, spvMSD, uses netCDF to save positional data (log spaced) for a simulation
of monodisperse cells in the SPV model (as compiled the .nc file will be saved to the current directory).
The file also contains a command line argument that allows for the system to be run under Brownian dynamics
rather than as a non-equilibrium active matter model. In any event, analyzing the saved positional data, for
instance by computing the mean squared displacement, can be used to confirm the correct operation of the
program. The included "Msd.sh" script will generate some data that can be analyzed and directly compared
with Fig. 1 of Bi et al. (Phys. Rev. X 6, 021011 (2016)), which is the cannonical reference for the SPV model.

The fourth program, Minimize.out, shows how to use the included FIRE minimizer to do simple energy minimization.

The fifth program, DynamicalMatrix.out, shows how to use the Eigen interface to compute the dynamical matrix of
Voronoi systems (with quadratic perimeter terms) and extract the eigenvalues/eignevectors. This code is rather
specialized, and not as general as one would ultimately like to see.

# Mac OS X Instructions

The following instructions for compiling cellGPU on Mac OSX  were contributed by Gonca Erdemci-Tandogan (https://goncaerdemci.wordpress.com/).
It does not include how to get Eigen (for the dynamical matrix diagonalization), but that is a header-only library.

Everything is most conveniently done using homebrew, the installation of which can be followed from https://brew.sh

## CGAL

brew install cgal

## netCDF

brew install homebrew/science/netcdf --with-cxx-compat

## CUDA

1. ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null ; brew install caskroom/cask/brew-cask 2> /dev/null

2. brew cask install cuda

3a. After cuda is installed, run "which nvcc". If this returns empty, you must export the path... 

3b. vim $HOME/.bashrc

3c. Add the following lines to the .bashrc and .bash_profile files:
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

doxygen: http://www.stack.nl/~dimitri/doxygen/download.html
