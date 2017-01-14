# INSTALLATION

A general makefile is included with the repository. To install on your system, update the CUDA_LIB,
CUDA_INC, LIB_CUDA, LIB_CGAL, LIB_NETCDF paths, and make sure the PATH and LD_LIBRARY_PATH
environment variables are appropriately set.
Create the /obj and /obj/cuobj directories, and from there, a simple "make" will do the trick.

# Requirements

The current iteration of the code was compiled using CUDA-8.0. The code has been tested with CUDA
versions as early as 6.5, and uses compute capability 3.5 devices and higher. It ought to work on
lower compute capability devices; compile without the "-arch=sm_35" flag to run on them

The SPV branch uses the CGAL library; this dependency can be removed, if necessary, by monkeying
with the code to run "fullTriangulation()" rather than "globalTriangulationCGAL()" in the relevant
spots. This is highly discouraged, and the code may be much less stable as a result. In any event,
CGAL-4.9 was used, which in turn requires up to date versions of the gmp and mpfr libraries.
The code was developed and tested against gmp-6.1.2 and mpfr-3.1.5.

The database class uses the netCDF-4 C++  library (tested on version 4.1.3).The dependency on netCDF can be removed by (1) not including any "Database" class, and (2) commenting out everything after the = sign in the LIB_NETCDF entry of the makefile

Documentation is maintained via doxygen, but is not required for compilation of the executables.

The makefile system is scheduled to be replaced, and will soon change to require cmake.

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads

CGAL: http://www.cgal.org/download.html

netCDF: http://www.unidata.ucar.edu/downloads/netcdf/index.jsp (be sure to get the C++ release, not the C release)

doxygen: http://www.stack.nl/~dimitri/doxygen/download.html
