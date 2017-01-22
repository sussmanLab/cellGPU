# Code snippets {#code}

These minimal samples show how to run default-initialized simulations. Please see the provided .cpp files in the main directory for more samples, or consult the documentation for more advanced usage.

## SPV code

With your main executable including the "spv2d.h" file, running a 2D SPV simulation is as simple as
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
