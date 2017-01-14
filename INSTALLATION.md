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

The database class uses the netCDF library, version 4.1.3.

Documentation is maintained via doxygen, and the makefile system will soon change to require cmake.

