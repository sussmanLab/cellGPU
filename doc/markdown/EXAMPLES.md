# Code snippets {#code}

These minimal samples show how to run default-initialized simulations. Please see the provided .cpp
files in the main directory and examples directory for more samples, or consult the documentation for more advanced usage.

# activeVertex.cpp

A simple interface that initializes and runs a reproducible simulation of the 2D active vertex model.
After being built with the included makefile, command line parameters can be passed in, controlling the
number of cells, the number of initialization time steps, the number of production-run time steps, the
time step size, the preferred cell energy and motility parameters, etc. This program can be used to
reproduce the timing information in the cellGPU paper.

# voronoi.cpp

The same thing, but for running the 2D SPV model

# minimize.cpp

Provides an example of using the FIRE minimizer to minimize either a 2D SPV or AVM system.

# runMakeDatabase.cpp

Provides an example of using the NetCDF database class to write snapshots of a simulation of the 2D
SPV model, using either active cell or overdamped Brownian dynamics

# cellDivision.cpp

Provides a simple example of both vertex and voronoi models with cell division activated

# dynMat.cpp

Provides an example of computing the dynamical matrix of a minimized voronoi model and using the
Eigen interface to diagonalize it.

