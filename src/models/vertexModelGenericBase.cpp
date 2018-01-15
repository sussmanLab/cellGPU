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
    fillGPUArrayWithVector(vertexCoordination,vertexCellNeighborNum);
    //initializeEdgeFlipLists();//NOTE THAT when this is written it must be added to remove cells and any other topology-changing functions

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
    ArrayHandle<int> h_vcnn(vertexCellNeighborNum,access_location::host,access_mode::read);
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
            };
        neighs = h_vcnn.data[vertex];
        for (int n = 0; n < neighs; ++n)
            {
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
    ArrayHandle<int> h_vcnn(vertexCellNeighborNum,access_location::host,access_mode::read);

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
            int vertexCoordination = h_vcnn.data[vidx];
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

/*!
Remove any cell whose index is in the list. Also discard any vertices which are now no longer part
of a cell. This then requires some laborious -- but straightforward -- reindexing and resizing of
GPUArrays
*/
void vertexModelGenericBase::removeCells(vector<int> cellIndices)
    {
    //figure out the cells and vertices left
    sort(cellIndices.begin(), cellIndices.end());
    vector<int> verticesToRemove;
    vector<int> newVertexNeighborNumber;
    copyGPUArrayData(vertexNeighborNum,newVertexNeighborNumber);
    
    {//array handle scope
    ArrayHandle<int> cvn(cellVertexNum);
    ArrayHandle<int> cv(cellVertices);
    for (int cc = 0; cc <cellIndices.size();++cc)
        {
        int cIdx = cellIndices[cc];
        int vNeighs = cvn.data[cellIndices[cc]];
        for (int vv = 0; vv < vNeighs; ++vv)
            {
            int vidx = cv.data[cellNeighborIndexer(vv,cIdx)];
            newVertexNeighborNumber[vidx] -= 1;
            if (newVertexNeighborNumber[vidx] <= 0)
                verticesToRemove.push_back(vidx);
            };
        };
    }//array handle scope
    sort(verticesToRemove.begin(),verticesToRemove.end());
    //Now, create a map of (old index, new index) for cells and vertices, with new index = -1 if it is to be removed
    //so, e.g., cellMap[100] is the new index of what used to be cell 100, or is -1 if it was to be removed
    //cellMapInverse is a vector of length new Ncells. cellMapInverse[10] gives the index of the cell that will be mapped to the 10th cell
    vector<int> cellMap(Ncells);
    vector<int> cellMapInverse(Ncells-cellIndices.size());
    vector<int> vertexMap(Nvertices);
    vector<int> vertexMapInverse(Nvertices-verticesToRemove.size());
    int idx = 0;
    int newidx = 0;
    int shift = 0;
    for (int cc = 0; cc < Ncells; ++cc)
        {
        int nextToRemove = cellIndices[idx];
        if(cc != nextToRemove)
            {
            cellMap[cc] = cc-shift;
            cellMapInverse[newidx] = cc;
            newidx +=1;
            }
        else
            {
            cellMap[cc] = -1;
            shift += 1;
            if(idx < cellIndices.size() -1)
                idx +=1;
            }
        };
    newidx=0;
    idx = 0;
    shift = 0;
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int nextToRemove = verticesToRemove[idx];
        if(vv != nextToRemove)
            {
            vertexMap[vv] = vv-shift;
            vertexMapInverse[newidx] = vv;
            newidx +=1;
            }
        else
            {
            vertexMap[vv] = -1;
            shift += 1;
            if(idx < verticesToRemove.size() -1)
                idx +=1;
            }
        };
    //Great... now we need to correct basically all of the data structures
    int NvOld = Nvertices;
    int NcOld = Ncells;
    Nvertices = NvOld - verticesToRemove.size();
    Ncells = NcOld - cellIndices.size();

    //first, let's handle the arrays that don't require any logic
    removeGPUArrayElement(vertexPositions,verticesToRemove);
    removeGPUArrayElement(vertexMasses,verticesToRemove);
    removeGPUArrayElement(vertexForces,verticesToRemove);
    removeGPUArrayElement(vertexVelocities,verticesToRemove);
    removeGPUArrayElement(displacements,verticesToRemove);

    removeGPUArrayElement(cellDirectors,cellIndices);
    removeGPUArrayElement(Motility,cellIndices);
    removeGPUArrayElement(AreaPeri,cellIndices);
    removeGPUArrayElement(AreaPeriPreferences,cellIndices);
    removeGPUArrayElement(cellForces,cellIndices);
    removeGPUArrayElement(cellMasses,cellIndices);
    removeGPUArrayElement(cellVelocities,cellIndices);
    removeGPUArrayElement(Moduli,cellIndices);
    removeGPUArrayElement(cellType,cellIndices);
    removeGPUArrayElement(cellPositions,cellIndices);

    //let's handle the vertex-based structures
    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);
    Index2D oldVNI = Index2D(vertexCoordinationMaximum,NvOld);
    GPUArray<int> newVertexNeighbors, newVertexNeighborNum, newVertexCellNeighbors, newVertexCellNeighborNum;
    newVertexNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    newVertexCellNeighbors.resize(vertexCoordinationMaximum*Nvertices);
    newVertexNeighborNum.resize(Nvertices);
    newVertexCellNeighborNum.resize(Nvertices);
    {//scope for array handles
    ArrayHandle<int> vn(newVertexNeighbors); ArrayHandle<int> vnOld(vertexNeighbors);
    ArrayHandle<int> vcn(newVertexCellNeighbors); ArrayHandle<int> vcnOld(vertexCellNeighbors);
    ArrayHandle<int> vnn(newVertexNeighborNum); ArrayHandle<int> vnnOld(vertexNeighborNum);
    ArrayHandle<int> vcnn(newVertexCellNeighborNum); ArrayHandle<int> vcnnOld(vertexCellNeighborNum);
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int oldVidx = vertexMapInverse[vv];
        //first, do the vertex neighbors of new vertex vv
        int neighs = vnnOld.data[oldVidx];
        int neighCur = 0;
        for (int nn = 0; nn < neighs; ++nn)
            {
            int neighbor = vnOld.data[oldVNI(nn,oldVidx)];
            if(vertexMap[neighbor] != -1)
                {
                vn.data[vertexNeighborIndexer(neighCur,vv)] = vertexMap[neighbor];
                neighCur +=1;
                };
            };
        vnn.data[vv] = neighCur;
        //now do the cells that vertex vv borders
        neighCur = 0;
        neighs = vcnnOld.data[oldVidx];
        for (int nn = 0; nn < neighs; ++nn)
            {
            int cellNeighbor = vcnOld.data[oldVNI(nn,oldVidx)];
            if(cellMap[cellNeighbor] != -1)
                {
                vcn.data[vertexNeighborIndexer(neighCur,vv)] = cellMap[cellNeighbor];
                neighCur += 1;
                };
            };
            vcnn.data[vv] = neighCur;
        };
    }//scope for array handles
    vertexNeighbors.swap(newVertexNeighbors);
    vertexCellNeighbors.swap(newVertexCellNeighbors);
    vertexNeighborNum.swap(newVertexNeighborNum);
    vertexCellNeighborNum.swap(newVertexCellNeighborNum);

    vertexForceSets.resize(vertexCoordinationMaximum*Nvertices);
    voroCur.resize(vertexCoordinationMaximum*Nvertices);
    voroLastNext.resize(vertexCoordinationMaximum*Nvertices);

    
    //Great! Now the cell-based lists

    cellNeighborIndexer = Index2D(vertexMax,Ncells);  
    Index2D oldCNI = Index2D(vertexMax,NcOld);
    GPUArray<int> newCellVertices,newCellVertexNum;
    newCellVertices.resize(vertexMax*Ncells);
    newCellVertexNum.resize(Ncells);
    {//scope for array handles
    ArrayHandle<int> cv(newCellVertices); ArrayHandle<int> cvOld(cellVertices);
    ArrayHandle<int> cvn(newCellVertexNum); ArrayHandle<int> cvnOld(cellVertexNum);
    for (int cc = 0; cc < Ncells; ++cc)
        {
        int oldCidx = cellMapInverse[cc];
        int neighs = cvnOld.data[oldCidx];
        int neighCur = 0;
        for (int nn = 0; nn < neighs; ++nn)
            {
            int neighbor = cvOld.data[oldCNI(nn,oldCidx)];
            if(vertexMap[neighbor] != -1)
                {
                cv.data[cellNeighborIndexer(neighCur,cc)] = vertexMap[neighbor];
                neighCur +=1;
                };
            };
        cvn.data[cc] = neighCur;
        };
    }//scope for array handles

    cellVertexNum.swap(newCellVertexNum);
    cellVertices.swap(newCellVertices);

    //handle the spatial sorting arrays
    vector<int> newitt, newtti, newidxtotag, newtagtoidx;
    newitt.resize(Ncells);newtti.resize(Ncells);newidxtotag.resize(Ncells);newtagtoidx.resize(Ncells);
    for (int cc = 0; cc < Ncells; ++cc)
        {
        int oldCidx = cellMapInverse[cc];
        newitt[cc] = cellMap[itt[oldCidx]];
        newtti[cc] = cellMap[tti[oldCidx]];
        newidxtotag[cc] = cellMap[idxToTag[oldCidx]];
        newtagtoidx[cc] = cellMap[tagToIdx[oldCidx]];
        };
    itt=newitt;
    tti=newtti;
    idxToTag=newidxtotag;
    tagToIdx=newtagtoidx;

    vector<int> newittv, newttiv, newidxtotagv, newtagtoidxv;
    newittv.resize(Nvertices);newttiv.resize(Nvertices);newidxtotagv.resize(Nvertices);newtagtoidxv.resize(Nvertices);
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int oldVidx = vertexMapInverse[vv];
        newittv[vv] = vertexMap[ittVertex[oldVidx]];
        newttiv[vv] = vertexMap[ttiVertex[oldVidx]];
        newidxtotagv[vv] = vertexMap[idxToTagVertex[oldVidx]];
        newtagtoidxv[vv] = vertexMap[tagToIdxVertex[oldVidx]];
        };
    ittVertex=newittv;
    ttiVertex=newttiv;
    idxToTagVertex=newidxtotagv;
    tagToIdxVertex=newtagtoidxv;
    
    //edgeFlipLists
    computeGeometry();
    };
