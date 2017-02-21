# Code snippets {#code}

These minimal samples show how to run default-initialized simulations. Please see the provided .cpp
files in the main directory for more samples, or consult the documentation for more advanced usage.

# activeVertex.cpp

A simple interface that initializes and runs a reproducible simulation of the 2D active vertex model.
After being built with the included makefile, command line parameters can be passed in, controlling the
number of cells, the number of initialization time steps, the number of production-run time steps, the
time step size, the preferred cell energy and motility parameters, etc. This program can be used to
reproduce the timing information in the cellGPU paper.

# voronoi.cpp

The same thing, but for running the 2D SPV model

# minimize.cpp

Provides an example of using the FIRE minimizer to minimize a 2D SPV system.

# runMakeDatabase.cpp

Provides an example of using the NetCDF database class to write snapshots of a simulation of the 2D
SPV model


## SPV code snippet

If you do not want to look at the above, just have your main executable including the "spv2d.h" file,
running a 2D SPV simulation is as simple as
```
SPV2D spv(numberOfCells,preferredArea,preferredPerimeter);
for (int timeStep = 0; timeStep < maximumTimeStep; ++timeStep)
    spv.performTimestep();
```
Of course, many more features can be accessed with setting functions.

## AVM code

Similarly, with your main executable including the "avm2d.h" file, running a 2D AVM simulation is done by
```
AVM2D avm(numberOfCells,preferredArea,preferredPerimeter);
for (int timeStep = 0; timeStep < maximumTimeStep; ++timeStep)
    avm.performTimestep();
```
