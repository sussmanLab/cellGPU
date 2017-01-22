# Basic overview of the project {#basicinfo}

## Classes of note

The following are a handful of classes that make up the most important operations and interfaces of cellGPU.

### General structures and analysis classes

* Simple2DCell -- a class that defines flat data structures for some of the most common features of
off-lattice cell models, such as cell positions, vertex positions, neighbor information, etc. Also
contains the functionality to perform spatial sorting of cells and vertices for better memory locality.

* Simple2DActiveCell -- a child of Simple2DCell class with data structures and functions common to many off-lattice cell models with active dynamics

* DelaunayLoc -- Calculates candidate 1-rings of particles by finding an enclosing polygon of nearby points and finding all points in the circumcircle of the point and any two consecutive vertices of that polygon.

### Cellular dynamics

* DelaunayMD -- A core engine that operates as described below in ''Basic idea.'' Can move particles
and update their underlying Delaunay triangulation
* SPV2D -- A child class of DelaunayMD that implements cell motion according to the 2D SPV model.
* AVM2D -- A child of Simple2DActiveCell that implements a simple 2D dynamic ("active") vertex model
where the topology changes via a simple rule for T1 transitions


## Basic idea of SPV hybrid operation

The following describes the basic operation of the DelaunayMD class
* (1) CPU STEP: If necessary (ie. after initialization, or after a timestep where a lot of neighbors
need to be updated) a CGAL (default) or Bowyer-Watson (non-standard) routine is called to completely
retriangulate the point set.
* (2) GPU STEP: The points are moved around in the periodic box, possibly based on forces computed
by the GPU.
* (3) GPU STEP: The GPU checks the circumcircle of every Delaunay triangle from the last timestep
(i.e., we check the connectivity of the old triangulation to see if anything needs to be updated).
A list of particles to fix is generated. If this list is of length zero, no memory copies take place.
* (4) CPU STEP: If needed, every particle that is flagged for fixing gets its neighbor list repaired
on the CPU. A call to DelaunayLoc finds the candidate 1-ring of that particle (a set of points from
which the true Delaunay neighbors are a strict subset), and CGAL (again, the default) is called to
reduce the candidate 1-ring to the true set of neighbors.
* (5) CPU/GPU: The new topology of the triangulation and associated data structures are updated, and
the cycle of (2)-(5) can repeat.

## Basic idea of AVM GPU-only operation

The following describes the basic operation of the active-vertex model class included. It is much
simpler than the SPV branch!
* (1) CPU STEP: Initialize a domain of cells in some way. Currently a CGAL triangulation of a random
point set is used.
* (2) GPU STEP: Compute the geometry of the cells, taking into account possible self-intersections
(in the active vertex model cells are not guaranteed to be convex).A
The points are moved around in the periodic box, possibly based on forces computed by the GPU.
* (3) GPU STEP: Compute the forces based on the current cell geometry.
* (4) GPU STEP: Move particles around based on forces and some activity
* (5) GPU: Check for any topological transitions. Update all data structures on the GPU, and then
the cycle of (2)-(5) can repeat.


