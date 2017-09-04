#ifndef deprecated_h
#define deprecated_h
/*! \file deprecated.h
A file that typedefs old class names, as a way to keep from breaking existing code during the
rapid-changes phase of development
*/
class VoronoiQuadraticEnergy;
typedef VoronoiQuadraticEnergy Voronoi2D;

class VoronoiQuadraticEnergyWithTension;
typedef  VoronoiQuadraticEnergyWithTension VoronoiTension2D;

class VertexQuadraticEnergy;
typedef VertexQuadraticEnergy AVM2D;

#endif
