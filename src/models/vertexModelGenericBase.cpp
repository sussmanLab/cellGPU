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
    printf("vertices = %i\n", Nvertices);
    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);
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

/*!
Compute the area and perimeter of every cell on the CPU
*/
void vertexModelGenericBase::computeGeometryCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vnn(vertexNeighborNum,access_location::host,access_mode::read);

    //compute the geometry for each cell
    for (int i = 0; i < Ncells; ++i)
        {
        int neighs = h_cvn.data[i];
//      Define the vertices of a cell relative to some (any) of its verties to take care of periodic boundaries
        Dscalar2 cellPos = h_v.data[h_cv.data[cellNeighborIndexer(neighs-2,i)]];
        Dscalar2 vlast, vcur,vnext;
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        //compute the vertex position relative to the cell position
        vlast.x=0.;vlast.y=0.0;
        int vidx = h_cv.data[cellNeighborIndexer(neighs-1,i)];
        Box->minDist(h_v.data[vidx],cellPos,vcur);
        for (int nn = 0; nn < neighs; ++nn)
            {
            //for easy force calculation, save the current, last, and next vertex position in the approprate spot.
            int forceSetIdx= -1;
            int vertexCoordination = h_vnn.data[vidx];
            for (int ff = 0; ff < vertexCoordination; ++ff)
                if(h_vcn.data[vertexNeighborIndexer(ff,vidx)] == i)
                    forceSetIdx = vertexNeighborIndexer(ff,vidx);
            vidx = h_cv.data[cellNeighborIndexer(nn,i)];
            Box->minDist(h_v.data[vidx],cellPos,vnext);
            //contribution to cell's area is
            // 0.5* (vcur.x+vnext.x)*(vnext.y-vcur.y)
            Varea += SignedPolygonAreaPart(vcur,vnext);
            Dscalar dx = vcur.x-vnext.x;
            Dscalar dy = vcur.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            //save vertex positions in a convenient form
            h_vc.data[forceSetIdx] = vcur;
            h_vln.data[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
            //advance the loop
            vlast = vcur;
            vcur = vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };  

