# CellGPU

CellGPU implements GPU-accelerated algorithms to simulate off-lattice models of cells. Its current
two main feature sets focus on a Voronoi-decomposition-based model of two-dimensional monolayers
and on a two-dimensional dynamical version of the vertex model. CellGPU was born out of DMS'
"DelGPU" and "VoroGuppy" projects, and the current class structure is still an outgrowth of that
(please see the contributing page for information on upcoming code refactoring and new planned
features!)

Information on installing the project and contributing to it is contained in the relevant
markdown files in the base directory. Documentation of the code is maintained via Doxygen... go
to the "/doc" directory, type "doxygen Doxyfile", and go from there. From the index of the doxygen
-generated html, see the Modules page for information on the CUDA kernels used in the GPU-based
routines.

As with many performance-seeking codes, there is a tension between optimized computational speed
and elegant code structure and organization. As a first pass this repository seeks to err slightly
on the side of optimiztion, particularly with reagrds to having very flat data structures for the
underlying degrees of freedom -- the vertices and centers of cells -- rather than a more natural
representation of vertices and cells as classes that carry around pointers or other references to
their properties. We reap the benefits of this when transferring and operating on data on the GPU.

## Classes of note

* cellListGPU -- makes cell lists using the GPU
* DelaunayNP -- Calculates the Delaunay triangulation in a non-periodic 2D domain (via naive Bowyer-Watson)
* DelaunayCGAL --Calculates the Delaunay triangulation in either a periodic or non-periodic 2D domain (via CGAL)
* DelaunayLoc -- Calculates candidate 1-rings of particles by finding an enclosing polygon of nearby points and finding all points in the circumcircle of the point and any two consecutive vertices of that polygon.
* DelaunayMD -- A core engine that operates as described above in ''Basic idea''
* SPV2D -- A child class of DelaunayMD that implements the 2D SPV model forces.
* AVM2D -- A separate class, independent of DelaunayMD, that reuses some of the ideas on the rest of the program to implement a simple 2D active vertex model

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

## Basic idea of AVM hybrid operation

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

## CITATIONS

The local ''test-and-repair'' part of the code is parallelized using an idea from Chen and Gotsman's ''Localizing the delaunay triangulation and its parallel implementation,'' [Transactions on Computational Science XX (M. L. Gavrilova, C.J.K. Tan, and B. Kalantari, eds.), Lecture Notes in Computer Science, vol. 8110, Springer Berlin Heidelberg, 2013, Extended abstract in ISVD 2012, pp. 24–31, pp. 39–55 (English)]. In particular, that paper points out a locality condition for the Delaunay neighborhood of a given point. Given a polygon formed by other vertices that encloses the target point, the possible set of Delaunay neighbors of the target point are those points contained in any of the circumcircles that can be formed by that point and consecutive vertices of the polygon).

There are two underlying routines for computing full Delaunay triangulation of non-periodic and periodoc point sets. In default operation of the code, the routines called are all part of the CGAL library, and that should be cited [at least CGAL, Computational Geometry Algorithms Library, http://www.cgal.org]. In less-ideal operations the user can call a naive $(O(N^{1.5}))$ Bowyer-Watson algorithm based off of Paul Bourke's Triangulate code: paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China)


## Directory structure

In this repository follows a simple structure. The main executable, voroguppy.cpp is in the base directory. Header files are in inc/, source files are in src/, and object files get put in obj/ (which is .gitignored, by default). A super-explicit makefile is used.

## Contributors

Daniel M. Sussman
