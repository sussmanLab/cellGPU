# Change log

### version 0.10

Refactor class structure... Simple2DCell --> Simple2DActiveCell --> {AVM2D or DelaunayMD --> SPV2D --> SPV2DTension}

### version 0.9

Add AVM2D class to implement a simple active vertex model. 

### version 0.8 

Implement separate tension terms between unlike cells in 2DSPV

### version 0.7 

GPU optimizations

### version 0.6 

Numerous bug fixes

### version 0.5

Remove references to Triangle...make CGAL the default triangulation library

### version 0.4

SPV2D class inherits from DelaunayMD and implements the 2D self-propelled voronoi model

### version 0.3

DelaunayMD class integrates functionality of DelaunayCheck with DelaunayLoc for local repairs

### version 0.2

DelaunayCheck class checks the validity of a proposed triangulation on either the CPU or GPU

### version 0.1 

CPU implementation of Chen and Gotsman's  localized Delaunay triangulation, comparisons with Shewchuck's Triangle program and a Bowyer-Watson implementation.
