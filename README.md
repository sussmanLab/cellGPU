#VoroGuppy
VoroGuppy (Voronoi decompositions on Graphics Processors) implements a parallelizable algorithm to calculate the Delaunay triangulation of a point set in a two-dimensional domain with periodic boundary conditions. The program can also be referred to as DelGPU (DELayed froGPUnch): Delaunay triangulation on a GPU. These names are aspirational more than actual, as this program will likely be purely serial for some time.


##CITATIONS
This code is parallelized around an idea from Chen and Gotsman's ``Localizing the delaunay triangulation and its parallel implementation,'' [Transactions on Computational Science XX (M. L. Gavrilova, C.J.K. Tan, and B. Kalantari, eds.), Lecture Notes in Computer Science, vol. 8110, Springer Berlin Heidelberg, 2013, Extended abstract in ISVD 2012, pp. 24–31, pp. 39–55 (English)]. In particular, that paper points out a locality condition for the Delaunay neighborhood of a given point. Given a polygon formed by other vertices that encloses the target point, the possible set of Delaunay neighbors of the target point are those points contained in any of the circumcircles that can be formed by that point and consecutive vertices of the polygon).

One of the underlying routines (for non-periodic systems) is a naive $(O(N^{1.5}))$ Bowyer-Watson algorithm based off of Paul Bourke's Triangulate code: paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China)

##File descriptions
* CPU-only files (mostly for testing purposes and algorithm design... but for small triangulations will be faster):
    * Delaunay1.h/cpp -- Defines DelaunayNP, a class to contruct the Delaunay triangulation of a local (non-periodic) set of points. Can easily get the local voronoi cell of a targeted Delaunay vertex. This is a purely serial class.
    * DelaunayLoc.h/cpp -- Defines a class that looks at a vertex, localizes the Delaunay triangulation to a possible set of vertices, then calls DelaunayNP to find the triangulation of the set and the voronoi cell of the vertex.
    * cell.h -- a class with a cell list
* GPU files:
* Assorted helper files:
    * structures.h -- contains classes corresponding to helpful structures (points, triangles, triangulations, local voronoi cells)
    * functions.h -- contains helpful functions, such as those correpsonding to circumcirle operations
    * box.h -- a class for computing periodic boundary condition distances
    * the ext_src directory contains Shewchuk's Triangle code
