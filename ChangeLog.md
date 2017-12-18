# Change log {#changelog}

### Changes in progress

* A directory to put common MD data analysis tools has been added

### version 0.7.1

* Support for non-square boxes... DelaunayCGAL calls currently implemented, but not DelaunayLoc
* Nose-Hoover thermostat-chain NVT simulations
* Starting transition away from Simple2DActiveCell class by using velocity GPUArrays in SPP updater

### version 0.7

* Cell death added. This removes a cell in Voronoi model, or does a T2 transition in vertex model
* BoxPtrs implemented...will eventually make different box shapes easier
* rationalize naming scheme of vertex and voronoi models
    * There are now vertexModelBase and voronoiModelBase classes
    * Derived classes of these largely just need to implement force laws
* include tree cleaning

### version 0.6.2

* Implement cell division in both vertex and Voronoi models
* Major bug fix in "quadratic" vertex model force computations

### version 0.6.1

* system-wide refactoring into simulation components: updaters, configurations, pieces, and a Simulation to tie them together
* added simple Brownian dynamics equation of motion
* added a (somewhat specialized) calculation of the quadratic voronoi model dynamical matrix

### version 0.6

* started refactoring class structures... beginning with equations of motion

### version 0.5.1

* added FIRE minimization algorithm
* Simplified interface to SPV2D and AVM2D; have all models implement virtual functions to provide a
common interface for MD

### version 0.5

* First major restructuring of class structure...

### cellGPU version 0.4

* The AVM2D class implements a simple active vertex model
* numerous bug fixes
* GPU optimizations

### version 0.3

* SPV2D class inherits from DelaunayMD; implements the 2D self-propelled voronoi model
* references to Triangle are removed; CGAL becomes the default triangulation library

### DelGPU version 0.2

* "DelGPU" implements the DelaunayCheck class to assess the validity of a proposed triangulation on
either the CPU or GPU; DelaunayMD provides a simple interface to an MD-like protocol

### VoroGuppy version 0.1

* "VoroGuppy" has a CPU implementation of Chen and Gotsman's localized Delaunay triangulation,
comparisons with Shewchuck's Triangle program and a Bowyer-Watson implementation.
