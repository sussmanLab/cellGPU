#DelGPU

DelGPU (''DELayed froGPUnch:Delaunay on GPUS) implements highly parallelizable algorithms to calculate the Delaunay triangulation of a point set in a two-dimensional domain with periodic boundary conditions. (The program can also be referred to as ''VoroGuppy: Voronoi decompositions on Graphics Processors,'' which is where the logo comes from.).

The primary engine -- the DelaunayMD class -- is a hybrid CPU/GPU algorithm intended to be used in ''molecular dynamics'' simulations where the particles are either Delaunay vertices or are derivable from them (as in the self-propelled Voronoi models of cells). It specializes in situations where the entire triangulation does not need to be recomputed at every time step, but rather only a small portion of the triangulation is to be repaired at a given time.

As an offshoot, the AVM class implements an active vertex model, reusing many of the ideas developed for the GPU-accelerated SPV simulation.

Documentation of the code is maintained via Doxygen... go to the "/doc" directory, type "doxygen Doxyfile", and go from there.


##Basic idea

The following describes the basic operation of the DelaunayMD class
* (1) CPU STEP: If necessary (ie. after initialization, or after a timestep where a lot of neighbors need to be updated) a CGAL (default) or Bowyer-Watson (non-standard) routine is called to completely retriangulate the point set.
* (2) GPU STEP: The points are moved around in the periodic box, possibly based on forces computed by the GPU.
* (3) GPU STEP: The GPU checks the circumcircle of every Delaunay triangle from the last timestep (i.e., we check the connectivity of the old triangulation to see if anything needs to be updated). A list of particles to fix is generated. If this list is of length zero, no memory copies take place.
* (4) CPU STEP: If needed, every particle that is flagged for fixing gets its neighbor list repaired on the CPU. A call to DelaunayLoc finds the candidate 1-ring of that particle (a set of points from which the true Delaunay neighbors are a strict subset), and CGAL (again, the default) is called to reduce the candidate 1-ring to the true set of neighbors.
* (5) CPU/GPU: The new topology of the triangulation and associated data structures are updated, and the cycle of (2)-(5) can repeat.

##Classes of note

* cellListGPU -- makes cell lists using the GPU
* DelaunayNP -- Calculates the Delaunay triangulation in a non-periodic 2D domain (via naive Bowyer-Watson)
* DelaunayCGAL --Calculates the Delaunay triangulation in either a periodic or non-periodic 2D domain (via CGAL)
* DelaunayLoc -- Calculates candidate 1-rings of particles by finding an enclosing polygon of nearby points and finding all points in the circumcircle of the point and any two consecutive vertices of that polygon.
* DelaunayMD -- A core engine that operates as described above in ''Basic idea''
* SPV2D -- A child class of DelaunayMD that implements the 2D SPV model forces.
* AVM2D -- A separate class, independent of DelaunayMD, that reuses some of the ideas on the rest of the program to implement a simple 2D active vertex model
##CURRENT LIMITATION

At the moment everything is optimized assuming the box is square. For this assumption to change, many edits would need to be made to the DelaunayCGAL and especially gpucell class. Also the grid class, and some changes in how higher-level classes interact with this objects.

##CITATIONS

The local ''test-and-repair'' part of the code is parallelized using an idea from Chen and Gotsman's ''Localizing the delaunay triangulation and its parallel implementation,'' [Transactions on Computational Science XX (M. L. Gavrilova, C.J.K. Tan, and B. Kalantari, eds.), Lecture Notes in Computer Science, vol. 8110, Springer Berlin Heidelberg, 2013, Extended abstract in ISVD 2012, pp. 24–31, pp. 39–55 (English)]. In particular, that paper points out a locality condition for the Delaunay neighborhood of a given point. Given a polygon formed by other vertices that encloses the target point, the possible set of Delaunay neighbors of the target point are those points contained in any of the circumcircles that can be formed by that point and consecutive vertices of the polygon).

There are two underlying routines for computing full Delaunay triangulation of non-periodic and periodoc point sets. In default operation of the code, the routines called are all part of the CGAL library, and that should be cited [at least CGAL, Computational Geometry Algorithms Library, http://www.cgal.org]. In less-ideal operations the user can call a naive $(O(N^{1.5}))$ Bowyer-Watson algorithm based off of Paul Bourke's Triangulate code: paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China)


##Directory structure

In this repository follows a simple structure. The main executable, voroguppy.cpp is in the base directory. Header files are in inc/, source files are in src/, and object files get put in obj/ (which is .gitignored, by default). A super-explicit makefile is used.

##Contributors

Daniel M. Sussman -- everything so far!
