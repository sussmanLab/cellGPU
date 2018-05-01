# Citations for cellGPU {#cite}

If you use cellGPU for a publication or project, please cite the main cellGPU paper:

(1) "cellGPU: massively parallel simulations of dynamic vertex models" Daniel M. Sussman; Computer Physics Communications, volume 219, pages 400-406, (2017)

Here are some additional citation to consider, according to what parts of the code you use and your
taste on how much to cite:

(2) Chen and Gotsman ''Localizing the delaunay triangulation and its parallel implementation,''
[Transactions on Computational Science XX (M. L. Gavrilova, C.J.K. Tan, and B. Kalantari, eds.),Lecture Notes in Computer Science, vol. 8110, Springer Berlin Heidelberg, 2013, Extended abstract in ISVD 2012, pp. 24–31, pp. 39–55 (English)]

The local ''test-and-repair'' part of the code used in the SPV branch is parallelized using an idea
from this paper. In particular, it points out a locality condition for the Delaunay neighborhood of a given point
(Given a polygon formed by other vertices that encloses the target point, the possible set of Delaunay
neighbors of the target point are those points contained in any of the circumcircles that can be
formed by that point and consecutive vertices of the polygon).

(3) CGAL,Computational Geometry Algorithms Library, http://www.cgal.org

The default triangulation library, used in the Voronoi branch of the code

(4) paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China)

Staying on the "Delaunay" branch, there are two underlying routines for computing full Delaunay
triangulation of non-periodic and periodoc point sets. In default operation of the code, the
routines called are all part of the CGAL library, and that should be cited. In less-ideal operations
the user can call a naive $(O(N^{1.5}))$ Bowyer-Watson algorithm based off of the Paul Bourke Triangulate
code: paulbourke.net/papers/triangulate (Pan-Pacific Computer Conference, Beijing, China). This is
never used by the code if you do not explicitly invoke it in the DelaunayLoc class.

(5) [E. Bitzek et al. Phys. Rev. Lett. 97, 170201 (2006)](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.170201)

One of the energy minimizer uses a straightforward implementation of the FIRE minimization algorithm,
which is described in the above paper.

(6) [G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and M. L. Klein; Mol. Phys. 87, 1117 (1996)](http://www.tandfonline.com/doi/abs/10.1080/00268979600100761)

The NoseHooverChainNVT class integrates the Nose-Hoover equations of motion with a chain of thermostats,
and does so using an update scheme that is explicitly time-reversible. The algorithm to do this is
described in Martyna et al., (see also the nice algorithmic pseudo-code in the Frenkel & Smit textbook)

(7) [J. Ramirez, S. K. Sukumaran, B. Vorselaars, and A. E. Likhtman; J. Chem. Phys. 133, 154103 (2010)](http://aip.scitation.org/doi/abs/10.1063/1.3491098)

The "autocorrelator" class in the analysis tools section is entirely based around this Ramirez et al. paper.
