#ifndef vertexGenericQuadraticEnergy_H
#define vertexGenericQuadraticEnergy_H

#include "vertexModelGeneric.h"

/*! \file vertexGenericQuadraticEnergy.h */
//!Implement a 2D vertex model, using kernels in \ref avmKernels
/*!
A simple class that implements a particular force law (from the quadratic vertex model functional)
in the context of a generic vertex model
*/
class VertexGenericQuadraticEnergy : public vertexModelGeneric
    {
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VertexGenericQuadraticEnergy(int n, bool reprod);

        //virtual functions that need to be implemented
        //!compute the geometry and get the forces
        virtual void computeForces();

        //!compute the quadratic energy functional
        virtual Dscalar computeEnergy();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeForcesCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeForcesGPU();
    };
#endif
