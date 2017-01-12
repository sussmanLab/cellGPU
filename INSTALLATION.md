# INSTALLATION

A general makefile is included with the repository. To install on your system, update the CUDA_LIB,
CUDA_INC, LIB_CUDA, LIB_CGAL, LIB_NETCDF paths, and make sure the PATH and LD_LIBRARY_PATH
environment variables are appropriately set.
Create the /obj and /obj/cuobj directories, and from there, a simple "make" will do the trick.

# Requirements

The current iteration of the code was compiled using CGAL-4.9 and CUDA-8.0. The code has been tested with CUDA versions as early as 6.5.

The database class uses the netCDF library, version 4.1.3.

Documentation is maintained via doxygen, and the makefile system will soon change to require cmake.

