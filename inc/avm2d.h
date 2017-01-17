#ifndef AVM_H
#define AVM_H

#include "std_include.h"
#include "Simple2DActiveCell.h"
#include "cu_functions.h"

/*!
A class that implements an active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.

This class is a child of the Simple2DCell class, which provides data structures like the positions of
cells, vertex positions, indices of vertices around each cell, cells around each vertex, etc.
*/
//!Implement a 2D active vertex model, using kernels in \ref avmKernels
class AVM2D : public Simple2DActiveCell
    {
    //public functions first...
    /*!
    \todo implement a spatial sorting scheme...needs to handle sorting both the vertices and the cell indices...
    */
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true,bool runSPVToInitialize=false);

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool initGPU = true,bool spvInitialize = false);

        //!Set the length threshold for T1 transitions
        void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(int n, bool spvInitialize = false);

        //!progress through the parts of a time step...simply an interface to the correct other procedure
        void performTimestep();
        //!progress through the parts of a time step on the CPU
        void performTimestepCPU();
        //!progress through the parts of a time step on the GPU
        void performTimestepGPU();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeGeometryGPU();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeForcesCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeForcesGPU();

        //!Displace vertices and rotate directors on the CPU
        void displaceAndRotateCPU();
        //!Displace vertices and rotate directors on the GPU
        void displaceAndRotateGPU();

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

        //!Get the cell position from the vertices on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellPositionsGPU();

    //protected functions
    protected:
        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();

        //!if the maximum number of vertices per cell increases, grow the cellVertices list
        void growCellVerticesList(int newVertexMax);

        //!spatially sort the *vertices* along a Hilbert curve for data locality
        void spatialVertexSorting();

        //utility functions
        //!For finding T1s on the CPU; find the set of vertices and cells involved in the transition
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

    //public member variables...most of these should eventually be protected
    public:
        /*!
        if vertexEdgeFlips[3*i+j]=1 (where j runs from 0 to 2), the the edge connecting verte i and vertex
        vertexNeighbors[3*i+j] has been marked for a T1 transition
        */
        //! flags that indicate whether an edge should be GPU-flipped (1) or not (0)
        GPUArray<int> vertexEdgeFlips;

        /*!
        vertexForceSets[3*i], vertexForceSets[3*i+1], and vertexForceSets[3*i+2] contain the contribution
        to the net force on vertex i due to the three cell neighbors of vertex i
        */
        //!an array containing the three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;

        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

    //protected variables
    protected:
        //! data structure to help with cell-vertex list
        GPUArray<int> growCellVertexListAssist;

        //! it is important to not flip edges concurrently, so this data structure helps flip edges sequentially
        GPUArray<int> vertexEdgeFlipsCurrent;
        //! data structure to help with not simultaneously trying to flip nearby edges
        GPUArray<int> finishedFlippingEdges;

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
    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabase;
    };
#endif
