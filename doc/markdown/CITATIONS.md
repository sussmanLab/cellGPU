# Citations for cellGPU {#cite}

The local ''test-and-repair'' part of the code used in the SPV branch is parallelized using an idea
from Chen and Gotsman's ''Localizing the delaunay triangulation and its parallel implementation,''
[Transactions on Computational Science XX (M. L. Gavrilova, C.J.K. Tan, and B. Kalantari, eds.),
Lecture Notes in Computer Science, vol. 8110, Springer Berlin Heidelberg, 2013, Extended abstract
in ISVD 2012, pp. 24–31, pp. 39–55 (English)]. In particular, that paper points out a locality
condition for the Delaunay neighborhood of a given point. Given a polygon formed by other vertices
that encloses the target point, the possible set of Delaunay neighbors of the target point are
those points contained in any of the circumcircles that can be formed by that point and consecutive
vertices of the polygon).

Also on the "Delaunay" branch, there are two underlying routines for computing full Delaunay
triangulation of non-periodic and periodoc point sets. In default operation of the code, the
routines called are all part of the CGAL library, and that should be cited [at least CGAL,
Computational Geometry Algorithms Library, http://www.cgal.org]. In less-ideal operations the user
can call a naive $(O(N^{1.5}))$ Bowyer-Watson algorithm based off of Paul Bourke's Triangulate
code: paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China)



