#ifndef AVM_H
#define AVM_H

#include "std_include.h"
#include "vertexModelBase.h"
//include spp dynamics for SPV-based initialization of configurations
#include "selfPropelledParticleDynamics.h"
#include "Simulation.h"

/*! \file avm2d.h */
//!Implement a 2D active vertex model, using kernels in \ref avmKernels
/*!
A class that implements a simple active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.
Only T1 transitions are currently implemented, and they occur whenever two vertices come closer
than a set threshold distance. All vertices are three-valent.

This class is a child of the vertexModelBase class, which provides data structures like the positions of
cells, vertex positions, indices of vertices around each cell, cells around each vertex, etc.
*/
class AVM2D : public vertexModelBase
    {
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool runSPVToInitialize=false);

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool spvInitialize = false);

        //!Set the length threshold for T1 transitions
        void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(bool spvInitialize = false);

        //virtual functions that need to be implemented
        //!compute the geometry and get the forces
        virtual void computeForces();

        //!update/enforce the topology
        virtual void enforceTopology();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeForcesCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeForcesGPU();

        //!Enforce CPU-only operation.
        void setCPU(bool global = true){GPUcompute = false;};

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

        //!spatially sort the *vertices* along a Hilbert curve for data locality
        virtual void spatialSorting();

    //protected functions
    protected:
        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();

        //!if the maximum number of vertices per cell increases, grow the cellVertices list
        void growCellVerticesList(int newVertexMax);

        //utility functions
        //!For finding T1s on the CPU; find the set of vertices and cells involved in the transition
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);
        //!Initialize the data structures for edge flipping...should also be called if Nvertices changes
        void initializeEdgeFlipLists();

    //public member variables...most of these should eventually be protected
    public:
        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

    //protected variables
    protected:
        //! data structure to help with cell-vertex list
        GPUArray<int> growCellVertexListAssist;

        //! data structure to help with not simultaneously trying to flip nearby edges
        GPUArray<int> finishedFlippingEdges;

    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabaseNetCDF;
    };
#endif
