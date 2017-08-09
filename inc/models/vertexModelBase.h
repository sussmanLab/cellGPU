#ifndef vertexModelBase_H
#define vertexModelBase_H

#include "std_include.h"
#include "Simple2DActiveCell.h"
#include "functions.h"

/*! \file vertexModelBase.h */
//!A class that can calculate many geometric and topological features common to vertex models
/*!
This class captures many energy-functional-independent features common to 2D vertex models of cells.
It can compute geometric features of cells, such as their area and perimeter; it knows that in vertex
models the degrees of freedom are the vertices (for returning the right vector of forces, moving
d.o.f. around, etc.); and it is where a geometric form of cell division is implemented.
 */

class vertexModelBase : public Simple2DActiveCell
    {
    public:
        //!In vertex models the number of degrees of freedom is the number of vertices
        virtual int getNumberOfDegreesOfFreedom(){return Nvertices;};

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> & displacements);

        //!return the forces
        virtual void getForces(GPUArray<Dscalar2> &forces){forces = vertexForces;};

        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<Dscalar2> & returnForces(){return vertexForces;};

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeGeometryGPU();

        //!Get the cell position from the vertices on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellPositionsGPU();

        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(vector<int> &parameters);

        /*!
        if vertexEdgeFlips[3*i+j]=1 (where j runs from 0 to 2), the the edge connecting vertex i and vertex
        vertexNeighbors[3*i+j] has been marked for a T1 transition
        */
        //! flags that indicate whether an edge should be GPU-flipped (1) or not (0)
        GPUArray<int> vertexEdgeFlips;
        //! it is important to not flip edges concurrently, so this data structure helps flip edges sequentially
        GPUArray<int> vertexEdgeFlipsCurrent;

        /*!
        vertexForceSets[3*i], vertexForceSets[3*i+1], and vertexForceSets[3*i+2] contain the contribution
        to the net force on vertex i due to the three cell neighbors of vertex i
        */
        //!an array containing the three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;

    protected:

    //reporting functions
    public:
        //!Handy for debugging T1 transitions...report the vertices owned by cell i
        void reportNeighborsCell(int i)
            {
            ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
            ArrayHandle<int> h_cv(cellVertices,access_location::host,access_mode::read);
            int cn = h_cvn.data[i];
            printf("Cell %i's neighbors:\n",i);
            for (int n = 0; n < cn; ++n)
                {
                printf("%i, ",h_cv.data[n_idx(n,i)]);
                }
            cout <<endl;
            };
    };

#endif
