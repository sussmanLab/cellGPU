This README.txt exists to satisfy journal submission requirements.

File names and brief descriptions:
activeVertex.cpp -- Can be compiled into an executable that generates timing information for the active vertex model
ChangeLog.md -- information on changes with each version of the code
INSTALLATION.md -- information on how to install cellGPU and its requirements
makefile -- a very explicit makefile that, given proper PATH and LD_LIBRARY_PATH specifications, can be used to compile the three .cpp files in the base directory into executables
LICENCE.md -- Open source licensing information
Msd.sh -- Runs spvMSD to create databases of SPV runs that can be used to reproduce data in Fig. 1 of Bi et al. (Phys. Rev. X 6, 021011 (2016)), which is the cannonical reference for the SPV model.
minimize.cpp -- Can be compiled into an executable demonstrating the use of the built-in FIRE energy minimization for either SPV or AVM systems
README.md -- The main page for the doxygen-generated documentation of cellGPU
runMakeDatabase.cpp -- Can be compiled into an executable that creates a database of snapshots from a self-propelled Voronoi model run.
Timing.sh -- A script that can run the executables compiled from voronoi.cpp or activeVertex.cpp to generate timing information of the code.
voronoi.cpp -- Can be compiled into an executable that generates timing information for the self-propelled Voronoi model
doc/* -- a directory that contains several files supporting doxygen-generated documentation of the repository

A list and brief description of all of the .h, .cuh, .cpp, and .cu files (in the inc/ and src/ directories)
is automatically maintained by doxygen documentation. It is also available on the public documentation,
which can be found at https://dmsussman.gitlab.io/cellGPUdocumentation

Installation instructions:
Please see the doxygen documentation at the above page for installation instructions, or see the INSTALLATION.md file
