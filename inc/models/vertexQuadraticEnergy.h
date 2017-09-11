#ifndef vertexQuadraticEnergy_H
#define vertexQuadraticEnergy_H

#include "vertexModelBase.h"

/*! \file vertexQuadraticEnergy.h */
//!Implement a 2D active vertex model, using kernels in \ref avmKernels
/*!
A class that implements a simple active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.

This class is a child of the vertexModelBase class, which provides data structures like the positions of
cells, vertex positions, indices of vertices around each cell, cells around each vertex, etc.
updates/enforces the topology according to vertexModelBase' T1 functions
*/
class VertexQuadraticEnergy : public vertexModelBase
    {
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VertexQuadraticEnergy(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool runSPVToInitialize=false);

        //virtual functions that need to be implemented
        //!compute the geometry and get the forces
        virtual void computeForces();

        //!compute the quadratic energy functional
        virtual Dscalar computeEnergy();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeForcesCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeForcesGPU();

    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabaseNetCDF;
    };
#endif
