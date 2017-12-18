# Code snippets {#code}

These minimal samples show how to run default-initialized simulations. Please see the provided .cpp
files in the main directory and examples directory for more samples, or consult the documentation for more advanced usage.

# Vertex.cpp

A simple interface that initializes and runs a simulation of the 2D vertex model.
After being built with the included makefile, command line parameters can be passed in, controlling the
number of cells, the number of initialization time steps, the number of production-run time steps, the
time step size, the preferred cell energy and motility parameters, etc. This program can be used to
reproduce the timing information in the cellGPU paper.

# voronoi.cpp

The same thing, but for running the 2D Voronoi model

# minimize.cpp

Provides an example of using the FIRE minimizer to minimize either a 2D Voronoi or AVM system.

# runMakeDatabase.cpp

Provides an example of using the NetCDF database class to write snapshots of a simulation of the 2D
Voronoi model, using either active cell or overdamped Brownian dynamics

# tensions.cpp

Provides a simple example of adding line tension terms to a Voronoi model

# vertexTensions.cpp

Provides a simple example of adding line tension terms to a 2D vertex model... see comments in this file
for caveats

# cellDivision.cpp

Provides a simple example of both vertex and voronoi models with cell division activated

# cellDeath.cpp

Provides a simple example of both vertex and voronoi models with cell death activated...
very similar to cellDivision, but particularly in the vertex model some extra care is needed

# dynMat.cpp

Provides an example of computing the dynamical matrix of a minimized voronoi model and using the
Eigen interface to diagonalize it.

# nvtVoronoi.cpp

Example setting up and using the NoseHooverChainNVT integrator. Nothing special
