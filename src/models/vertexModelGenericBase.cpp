#define ENABLE_CUDA

#include "vertexModelGenericBase.h"
#include "DelaunayCGAL.h" 
#include "vertexModelGenericBase.cuh"
/*! \file vertexModelGenericBase.cpp */

/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void vertexModelGenericBase::initializeVertexGenericModelBase(int n)
    {
    //call initializer chain...sets Ncells = n
    //Note that this, by default, uses a random voronoi construction with every vertex being three-fold coordinated
    initializeSimpleVertexModelBase(n);
    vertexCoordinationMaximum = 3;
    vertexNeighborIndexer = Index2D(vertexMax,Nvertices);
    vector<int> vertexCoordination(Nvertices,3);
    fillGPUArrayWithVector(vertexCoordination,vertexNeighborNum);
    //initializeEdgeFlipLists();

    //growCellVertexListAssist.resize(1);
    //ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::overwrite);
    //h_grow.data[0]=0;
    };

/*!
When a transition increases the maximum coordination of the vertices in the system,
call this function first to copy over the vertexNeighbors and vertexCellNeighbors structures
and resize the forceSets, voroCur, and voroLastNext structures
 */
void vertexModelGenericBase::growVertexNeighborLists(int newCoordinationMax)
    {
    cout << "maximum vertex coordination grew from " << vertexCoordinationMaximum << " to " << newCoordinationMax << endl;
    vertexCoordinationMaximum = newCoordinationMax;

    Index2D old_indexer = vertexNeighborIndexer;
    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);

    GPUArray<int> newVertexNeighbors;
    GPUArray<int> newVertexCellNeighbors;
    newVertexNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    newVertexCellNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    
    {//scope for array handles
    ArrayHandle<int> h_vnn(vertexNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn_old(vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn_old(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn(newVertexNeighbors,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vcn(newVertexCellNeighbors,access_location::host,access_mode::overwrite);

    for (int vertex = 0; vertex < Nvertices; ++vertex)
        {
        int neighs = h_vnn.data[vertex];
        for (int n = 0; n < neighs; ++n)
            {
            h_vn.data[vertexNeighborIndexer(n,vertex)] = h_vn_old.data[old_indexer(n,vertex)];
            h_vcn.data[vertexNeighborIndexer(n,vertex)] = h_vcn_old.data[old_indexer(n,vertex)];
            };
        };
    };//scope for array handles
    vertexNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    vertexNeighbors.swap(newVertexNeighbors);
    vertexCellNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    vertexCellNeighbors.swap(newVertexCellNeighbors);

    //resize per-vertex-coordination lists
    vertexForceSets.resize(vertexCoordinationMaximum*Nvertices);
    voroCur.resize(vertexCoordinationMaximum*Nvertices);
    voroLastNext.resize(vertexCoordinationMaximum*Nvertices);
    };
