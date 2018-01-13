#ifndef simpleVertexModelBase_H
#define simpleVertexModelBase_H

#include "Simple2DActiveCell.h"
#include "Simulation.h"

/*! \file simpleVertexModelBase.h */
//!A class that can calculate many geometric and topological features common to all vertex models
/*!
This class captures many energy-functional-independent features common to 2D vertex models of cells.
It can compute geometric features of cells, such as their area and perimeter; it knows that in vertex
models the degrees of freedom are the vertices (for returning the right vector of forces, moving
d.o.f. around, etc.); and it is where a geometric form of cell division is implemented.

Vertex models have to maintain the topology of the cell network by hand, so child classes need to
implement not only an energy functional (and corresponding force law), but also rules for topological
transitions. This base class just handles the functions that are common to all vertex models studied.
In particular, the functions here must not care whether all vertices are three-fold coordinate or not.
 */

class simpleVertexModelBase : public Simple2DActiveCell
    {
    public:
        //!Take care of some simple data structure initialization
        void initializeSimpleVertexModelBase(int n);

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> & displacements,Dscalar scale = 1.);
        //!In vertex models the number of degrees of freedom is the number of vertices
        virtual int getNumberOfDegreesOfFreedom(){return Nvertices;};
        //!spatially sort the *vertices* along a Hilbert curve for data locality
        virtual void spatialSorting();

        //!return the forces
        virtual void getForces(GPUArray<Dscalar2> &forces){forces = vertexForces;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<Dscalar2> & returnForces(){return vertexForces;};
        //!return a reference to the GPUArray of the current velocities
        virtual GPUArray<Dscalar2> & returnVelocities(){return vertexVelocities;};
        //!return a reference to the GPUArray of the current positions
        virtual GPUArray<Dscalar2> & returnPositions(){return vertexPositions;};
        //!return a reference to the GPUArray of the current masses
        virtual GPUArray<Dscalar> & returnMasses(){return vertexMasses;};
        //!Set the length threshold for T1 transitions
        virtual void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};
        //!Enforce CPU-only operation.
        void setCPU(bool global = true){GPUcompute = false;};

        //!Call the CPU or GPU getCellCentroids function
        void getCellCentroids();
        //!Get the cell position from the vertices on the CPU
        void getCellCentroidsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellCentroidsGPU();
        //!Call the CPU or GPU getCellPositions function
        void getCellPositions();
        //!Get the cell position from the average vertex position on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the average vertex position on the GPU
        void getCellPositionsGPU();

        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

        /*!
        vertexForceSets[3*i], vertexForceSets[3*i+1], and vertexForceSets[3*i+2] contain the contribution
        to the net force on vertex i due to the three cell neighbors of vertex i
        If the model is not always three-fold coordinated, will be indexed by a new indexer
        */
        //!an array containing (typically) three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;
    protected:
        //!if the maximum number of vertices per cell increases, grow the cellVertices list
        void growCellVerticesList(int newVertexMax);

    };
#endif
