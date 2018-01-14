#ifndef vertexModelGenericBase_H
#define vertexModelGenericBase_H

#include "simpleVertexModelBase.h"

/*! \file vertexModelGenericBase.h */
//!A class that allows vertices of arbitrary coordination number
/*!
This class lays the groundwork for simulating vertex models in which not every vertex is three-fold
coordinated.
*/

class vertexModelGenericBase : public simpleVertexModelBase
    {
    public:
        //!Initialize vertexModelBase,  prepare data structures
        void initializeVertexGenericModelBase(int n);

    /*
        //!Compute the geometry (area & perimeter) of the cells on the CPU
        virtual void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        virtual void computeGeometryGPU();

        //!"Remove" a cell...This function will delete a cell but leave its vertices (as long as the vertex is part of at least one cell...useful for creating open boundaries
        virtual void removeCell(int cellIndex);
        

        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(const vector<int> &parameters,const vector<Dscalar> &dParams = {});

        //!Kill the indexed cell...cell must have only three associated vertices
        virtual void cellDeath(int cellIndex);

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

        //!update/enforce the topology, performing simple T1 transitions
        virtual void enforceTopology();
    */

    protected:
        //!The largest highet coordination number of any vertex
        int vertexCoordinationMaximum;
        //!The number of vertices that vertex[i] is connected to
        GPUArray<int> vertexNeighborNum;
        //!A 2dIndexer for computing where in the GPUArray to look for a given vertex's vertex neighbors (or cell neighbors)
        /*!
        So, for instance, the kth vertex or cell neighbor of vertex i can ve accessed by:
        vertexNeighbors[vertexNeighborIndexer(k,i)];
        vertexCellNeighbors[vertexCellNeighborIndexer(k,i)];
        The maximum index that should be accessed in this way is given by vertexNeighborNum[i];
        */
        Index2D vertexNeighborIndexer;

        //!if the maximum vertex coordination increases, grow the vertexNeighbor and force set lists
        void growVertexNeighborLists(int newCoordinationMax);

    /*
        //!Initialize the data structures for edge flipping...should also be called if Nvertices changes
        void initializeEdgeFlipLists();

        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();

        //!For finding T1s on the CPU; find the set of vertices and cells involved in the transition
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

    */
    };
#endif
