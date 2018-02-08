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
        void initializeVertexModelGenericBase(int n);

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        virtual void computeGeometryCPU();


        //!"Remove" cells whose index matches those in the vector...This function will delete a cell but leave its vertices (as long as the vertex is part of at least one cell...useful for creating open boundaries
        virtual void removeCells(vector<int> cellIndices);
        //!Kill the indexed cell...cell can have any number of vertices
        virtual void cellDeath(int cellIndex);
        
        //!Merge a number of vertices into a single vertex...this construction means T2s are easy
        virtual void mergeVertices(vector<int> verticesToMerge);
        //!Take a vertex and divide it into two vertices
        virtual void splitVertex(int vertexIndex, Dscalar separation, Dscalar theta);

        //!spatially sort the *vertices* along a Hilbert curve for data locality...cannot call underlying routines!
        virtual void spatialSorting(){};
        //NEED TO WRITE THE ABOVE FUNCTION

/*
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        virtual void computeGeometryGPU();



    */
    /*
        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(const vector<int> &parameters,const vector<Dscalar> &dParams = {});


        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

        //!update/enforce the topology, performing simple T1 transitions
        virtual void enforceTopology();
    */

//    protected:




        //!Eventually we will be including vertex-edge merging events; to make this efficient we will maintain a separate edge structure
        //GPUArray<EDGE> edgeList;
        //!The largest  coordination number of any vertex
        int vertexCoordinationMaximum;
        //!The number of vertices that vertex[i] is connected to
        GPUArray<int> vertexNeighborNum;
        //!The number of cells that vertex[i] borders
        GPUArray<int> vertexCellNeighborNum;
        //!A 2dIndexer for computing where in the GPUArray to look for a given vertex's vertex neighbors (or cell neighbors)
        /*!
        So, for instance, the kth vertex or cell neighbor of vertex i can ve accessed by:
        vertexNeighbors[vertexNeighborIndexer(k,i)];
        vertexCellNeighbors[vertexCellNeighborIndexer(k,i)];
        The maximum index that should be accessed in this way is given by vertexNeighborNum[i];
        Note that this indexer will be used for both vertexNeighborNum and vertexCellNeighborNum,
        even though when there are open boundaries they may start to diverge
        */
        Index2D vertexNeighborIndexer;

        //!if the maximum vertex coordination increases, grow the vertexNeighbor and force set lists
        void growVertexNeighborLists(int newCoordinationMax);
        //!If the number of vertices changes, per-coordination-number lists should be resized
        void resizePerCoordinationLists();

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

        //The following are data structures that help manage topology after removals
        //!cellMap[i] is the index of what was cell i before a removal, or is -1 if it was removed
        vector<int> cellMap;
        //!cellMapInverse[i] gives the index of the cell that was mapped to the ith current cell
        vector<int> cellMapInverse;
        //!Same as cellMap, but for vertices
        vector<int> vertexMap;
        //!Same as cellMapInverse, but for vertices
        vector<int> vertexMapInverse;

        //!Reindex the cell sorting arrays based on cellMap an cellMapInverse
        void remapCellSorting();
        //!Reindex the vertex sorting arrays based on vertexMap an vertexMapInverse
        void remapVertexSorting();
    };
#endif
