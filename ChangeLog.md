# Change log {#changelog}

### Changes in progress

* added simple Brownian dynamics equation of motion

### version 0.6

* refactored code to implement simpleEquationOfMotion and child classes.

### version 0.5.1

* added FIRE minimization algorithm
* Simplified interface to SPV2D and AVM2D; have all models implement virtual functions to provide a
common interface for MD

### version 0.5

* Refactor class structure... Simple2DCell --> Simple2DActiveCell --> AVM2D and Simple2DActiveCell
--> DelaunayMD --> SPV2D

### version 0.4

* The AVM2D class implements a simple active vertex model
* numerous bug fixes
* GPU optimizations

### version 0.3

* SPV2D class inherits from DelaunayMD; implements the 2D self-propelled voronoi model
* references to Triangle are removed; CGAL becomes the default triangulation library

### version 0.2

* "DelGPU" implements the DelaunayCheck class to assess the validity of a proposed triangulation on
either the CPU or GPU; DelaunayMD provides a simple interface to an MD-like protocol

### version 0.1

* "VoroGuppy" has a CPU implementation of Chen and Gotsman's  localized Delaunay triangulation,
comparisons with Shewchuck's Triangle program and a Bowyer-Watson implementation.
