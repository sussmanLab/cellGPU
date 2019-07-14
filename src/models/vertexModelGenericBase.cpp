#define ENABLE_CUDA

#include "vertexModelGenericBase.h"
#include "DelaunayCGAL.h"
#include "vertexModelGenericBase.cuh"
/*! \file vertexModelGenericBase.cpp */

/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void vertexModelGenericBase::initializeVertexModelGenericBase(int n)
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
    //initialize the maps and inverses to lists from 0 to Ncells-1 (or Nvertices - 1)
    cellMap = itt;
    cellMapInverse=itt;
    vertexMap = ittVertex;
    vertexMapInverse = ittVertex;
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

    copyReIndexed2DGPUArray(vertexNeighbors,old_indexer,vertexNeighborIndexer);
    copyReIndexed2DGPUArray(vertexCellNeighbors,old_indexer,vertexNeighborIndexer);

    //resize per-vertex-coordination lists
    resizePerCoordinationLists();
    };

/*!
Whenever vertexCoordinationMaximum or Nvertices changes, the per-vertex-coordination lists are resized
*/
void vertexModelGenericBase::resizePerCoordinationLists()
    {
    vertexForceSets.resize(vertexCoordinationMaximum*Nvertices);
    voroCur.resize(vertexCoordinationMaximum*Nvertices);
    voroLastNext.resize(vertexCoordinationMaximum*Nvertices);
    if(vertexNeighborNum.getNumElements() < Nvertices)
        {
        int vToAdd = Nvertices-vertexNeighborNum.getNumElements();
        growGPUArray(vertexCellNeighborNum,vToAdd);
        growGPUArray(vertexNeighborNum,vToAdd);
        growGPUArray(vertexCellNeighbors,vToAdd*vertexCoordinationMaximum);
        growGPUArray(vertexNeighbors,vToAdd*vertexCoordinationMaximum);
        }
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
    if(cellIndices.size()<1)
        return;
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
    createIndexMapAndInverse(cellMap,cellMapInverse,cellIndices,Ncells);
    createIndexMapAndInverse(vertexMap,vertexMapInverse,verticesToRemove,Nvertices);

    //Great... now we need to correct basically all of the data structures
    int NvOld = Nvertices;
    int NcOld = Ncells;
    Nvertices = NvOld - verticesToRemove.size();
    Ncells = NcOld - cellIndices.size();

    //first, let's handle the arrays that don't require any logic
    if(verticesToRemove.size()>0)
        {
        removeGPUArrayElement(vertexPositions,verticesToRemove);
        removeGPUArrayElement(vertexMasses,verticesToRemove);
        removeGPUArrayElement(vertexForces,verticesToRemove);
        removeGPUArrayElement(vertexVelocities,verticesToRemove);
        removeGPUArrayElement(displacements,verticesToRemove);
        }
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

    resizePerCoordinationLists();

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
    remapCellSorting();
    remapVertexSorting();

    //edgeFlipLists
    computeGeometry();
    };

/*!
Take a vector of vertex indices and merge them into a single vertex, handling all of the data structure
updates necessary.
Of note, this will leave a single vertex at the position of the lowest-indexed vertex in the set of
vertices that is to be merged.
*/
void vertexModelGenericBase::mergeVertices(vector<int> verticesToMerge)
    {
    //for convenience, let's decide to keep the lowest-indexed vertex...for super-convenience, we'll sort the vector
    sort(verticesToMerge.begin(),verticesToMerge.end());

    int vertexIdx = verticesToMerge[0];
    //create vertexMap and vertexMapIndex
    vector<int> verticesToRemove(verticesToMerge.begin()+1,verticesToMerge.end());
    createIndexMapAndInverse(vertexMap,vertexMapInverse,verticesToRemove,Nvertices);

    //create a list of the new vertex and cell neighbors of the merged vertex...
    vector<int> vNeighbors, cNeighbors;
    vNeighbors.reserve(vertexCoordinationMaximum*verticesToMerge.size());
    cNeighbors.reserve(vertexCoordinationMaximum*verticesToMerge.size());
    {//array handle scope
    ArrayHandle<int> vn(vertexNeighbors);
    ArrayHandle<int> vnn(vertexNeighborNum);
    ArrayHandle<int> vcn(vertexCellNeighbors);
    ArrayHandle<int> vcnn(vertexCellNeighborNum);
    for (int vv = 0; vv < verticesToMerge.size(); ++vv)
        {
        int vidx = verticesToMerge[vv];
        int cneigh = vcnn.data[vidx];
        for (int nn = 0; nn < cneigh; ++nn)
            {
            int neighbor = vcn.data[vertexNeighborIndexer(nn,vidx)];
            cNeighbors.push_back(neighbor);
            };
        int vneigh = vnn.data[vidx];
        for (int nn = 0; nn < vneigh; ++nn)
            {
            int neighbor = vn.data[vertexNeighborIndexer(nn,vidx)];
            if (vertexMap[neighbor] != -1 && neighbor != vertexIdx)
                {
                vNeighbors.push_back(neighbor);
                };
            };
        };
    };
    removeDuplicateVectorElements(vNeighbors);
    int vNeighNum = vNeighbors.size();
    removeDuplicateVectorElements(cNeighbors);
    int cNeighNum = cNeighbors.size();

    if(vNeighNum > vertexCoordinationMaximum)
        {
        growVertexNeighborLists(vNeighNum);
        };

    //arrays that don't require logic
    removeGPUArrayElement(vertexPositions,verticesToRemove);
    removeGPUArrayElement(vertexMasses,verticesToRemove);
    removeGPUArrayElement(vertexForces,verticesToRemove);
    removeGPUArrayElement(vertexVelocities,verticesToRemove);
    removeGPUArrayElement(displacements,verticesToRemove);

    //vertex neighbor arrays: the merged vertex will be neighbors with all of the vertices and cells of the original vertices
    {//array handles
    ArrayHandle<int> h_vn(vertexNeighbors,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vnn(vertexNeighborNum,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vcnn(vertexCellNeighborNum,access_location::host,access_mode::readwrite);

    //edit the vertex and cell neighbors of the merged vertex
    h_vnn.data[vertexIdx] = vNeighNum;
    h_vcnn.data[vertexIdx] = cNeighNum;
    for (int nn = 0; nn < vNeighNum; ++nn)
        h_vn.data[vertexNeighborIndexer(nn,vertexIdx)] = vNeighbors[nn];
    for (int nn = 0; nn < cNeighNum; ++nn)
        h_vcn.data[vertexNeighborIndexer(nn,vertexIdx)] = cNeighbors[nn];
    //edit the vertex neighbors of vNeighbors...if a vNeighbor neighbored more than one merged vertex,
    //only add the merged vertex once
    for (int vv = 0; vv < vNeighNum; ++vv)
        {
        int vidx = vNeighbors[vv];
        int neigh = h_vnn.data[vidx];
        vector<int> VN; VN.reserve(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            int v2 = h_vn.data[vertexNeighborIndexer(nn,vidx)];
            if(vertexMap[v2] == -1)
                VN.push_back(vertexIdx);
            else
                VN.push_back(v2);
            };
        removeDuplicateVectorElements(VN);
        h_vnn.data[vidx] = VN.size();
        for (int nn = 0; nn < VN.size(); ++nn)
            h_vn.data[vertexNeighborIndexer(nn,vidx)] = VN[nn];
        };

    //account for the shift in vertex indices everywhere
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int neigh= h_vnn.data[vv];
        int newNeigh = 0;
        for (int nn = 0; nn < neigh; ++nn)
            {
            int vidx = h_vn.data[vertexNeighborIndexer(nn,vv)];
            if(vertexMap[vidx] != -1)
                {
                h_vn.data[vertexNeighborIndexer(newNeigh,vv)] = vertexMap[vidx];
                newNeigh += 1;
                };
            };
        h_vnn.data[vv] = newNeigh;
        };

    }//array handles

    removeGPUArrayElement(vertexNeighborNum,verticesToRemove);
    removeGPUArrayElement(vertexCellNeighborNum,verticesToRemove);

    vector<int> vinds; vinds.resize(vertexCoordinationMaximum*verticesToRemove.size());
    int vTempIdx=0;
    for (int vv = 0; vv < verticesToRemove.size();++vv)
        for (int nn = 0; nn < vertexCoordinationMaximum; ++nn)
            {
            vinds[vTempIdx] = vertexNeighborIndexer(nn,verticesToRemove[vv]);
            vTempIdx+=1;
            };
    removeGPUArrayElement(vertexNeighbors,vinds);
    removeGPUArrayElement(vertexCellNeighbors,vinds);

    //cell-based arrays: each cell that neighbored any of the merged vertices is neighbors only with the remaining vertex
    {//array handle
    ArrayHandle<int> cv(cellVertices);
    ArrayHandle<int> cvn(cellVertexNum);
    //first, shift the indices of all of the cell vertices for uninvolved cells
    sort(cNeighbors.begin(),cNeighbors.end());
    int cNidx = 0;
    for (int cc = 0; cc < Ncells; ++cc)
        {
        if(cc == cNeighbors[cNidx])
            {
            if (cNidx < cNeighbors.size()-1)
                cNidx += 1;
            continue;
            };
        int neighs = cvn.data[cc];
        for (int vv = 0; vv < neighs; ++vv)
            {
            int vidx = cv.data[cellNeighborIndexer(vv,cc)];
            cv.data[cellNeighborIndexer(vv,cc)] = vertexMap[vidx];
            };
        };
    //now, handle the cells which neighbored one of the merged vertices
    for (int cc  = 0; cc < cNeighbors.size(); ++cc)
        {
        int cidx = cNeighbors[cc];
        int neighs = cvn.data[cidx];
        vector<int> CV; CV.reserve(neighs);
        for (int vv = 0; vv < neighs; ++vv)
            {
            int vidx = cv.data[cellNeighborIndexer(vv,cidx)];
            //edit in place if the vertex still exists
            if(vertexMap[vidx] == -1)
                CV.push_back(vertexIdx);
            else
                CV.push_back(vertexMap[vidx]);
            };
        removeDuplicateVectorElements(CV);
        for (int vv = 0; vv < CV.size(); ++vv)
            cv.data[cellNeighborIndexer(vv,cidx)] = CV[vv];
        cvn.data[cidx] = CV.size();
        };
    }//array handle

    //resize remaining arrays
    Nvertices = Nvertices - verticesToRemove.size();
    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);

    //resize per-vertex-coordination lists
    resizePerCoordinationLists();
    //update vertex sorting arrays
    remapVertexSorting();
    };

/*!
This function inserts a new vertex in between two existing vertices.
All relevant data structures are updated.
*/
void vertexModelGenericBase::subdivideEdge(int vertexIndex1, int vertexIndex2)
    {
    //before indices change, what are the (one or two) cells that share this edge?

    //what are the (possibly one or two) cells that have both v1 and v2 as vertices?
    vector<int> cellsInvolved;
    vector<int> cellsVertexNumbers;
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vcnn(vertexCellNeighborNum,access_location::host,access_mode::readwrite);
    int cNeighs1 = h_vcnn.data[vertexIndex1];
    vector<int> cellsToTest(cNeighs1);
    for (int ii = 0; ii < cNeighs1; ++ii)
        cellsToTest[ii] = h_vcn.data[vertexNeighborIndexer(ii,vertexIndex1)];
    removeDuplicateVectorElements(cellsToTest);
    //which of those cells also have v2?
    ArrayHandle<int> cv(cellVertices);
    ArrayHandle<int> cvn(cellVertexNum);
    for (int cc = 0; cc < cellsToTest.size(); ++cc)
        {
        int cIdx = cellsToTest[cc];
        int vNeighs = cvn.data[cIdx];
        for (int vv = 0; vv < vNeighs; ++vv)
            if(cv.data[cellNeighborIndexer(vv,cIdx)] == vertexIndex2)
                {
                cellsInvolved.push_back(cIdx);
                cellsVertexNumbers.push_back(vNeighs);
                printf("cell to test: %i \n", cIdx);
                }
        }
    }//end array handle scope

    int newVertexIndex = Nvertices;
    Nvertices += 1;
    //add a new index for the spatial sorters
    ittVertex.push_back(newVertexIndex);
    ttiVertex.push_back(newVertexIndex);
    tagToIdxVertex.push_back(newVertexIndex);
    idxToTagVertex.push_back(newVertexIndex);
    //resize per-vertex-coordination lists
    resizePerCoordinationLists();

    //take care of the per-vertex lists
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    growGPUArray(vertexPositions,1);
    growGPUArray(vertexMasses,1);
    growGPUArray(vertexVelocities,1);

    //find the new position
    Dscalar2 newVertexPos;
    {
    ArrayHandle<Dscalar2> velocities(vertexVelocities);
    ArrayHandle<Dscalar2> positions(vertexPositions);
    Dscalar2 zero = make_Dscalar2(0.0,0.0);
    velocities.data[newVertexIndex] = zero;
    Dscalar2 disp;
    Box->minDist(positions.data[vertexIndex1],positions.data[vertexIndex2],disp);
    newVertexPos = positions.data[vertexIndex2] + 0.5*disp;
printf("%f %f\t %f %f\n",disp.x,disp.y,newVertexPos.x,newVertexPos.y);
    Box->putInBoxReal(newVertexPos);
    positions.data[newVertexIndex] = newVertexPos;
    }//end array handle scope

    //update vertex connectivity
    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);
    Index2D oldVNI = Index2D(vertexCoordinationMaximum,Nvertices-1);

    //vertex neighbors and vertex neighbor num
    {//scope for array handles
    ArrayHandle<int> vn(vertexNeighbors);
    ArrayHandle<int> vnn(vertexNeighborNum);

    //First, rewire the vertices formerly connected by an edge
    int neighs1 = vnn.data[vertexIndex1];
    for (int nn = 0; nn < neighs1; ++nn)
        {
        int otherIdx = vn.data[oldVNI(nn,vertexIndex1)];
        if (otherIdx == vertexIndex2)
            vn.data[oldVNI(nn,vertexIndex1)] = newVertexIndex;
        };
    int neighs2 = vnn.data[vertexIndex2];
    for (int nn = 0; nn < neighs2; ++nn)
        {
        int otherIdx = vn.data[oldVNI(nn,vertexIndex2)];
        if (otherIdx == vertexIndex1)
            vn.data[oldVNI(nn,vertexIndex2)] = newVertexIndex;
        };
    //...and give the new vertex its two Neighbors
    vnn.data[newVertexIndex] = 2;
    vn.data[vertexNeighborIndexer(0,newVertexIndex)] = vertexIndex1;
    vn.data[vertexNeighborIndexer(1,newVertexIndex)] = vertexIndex2;
    }


    //update cell vertex number...

    //first, do sizes need to be redone?
    int newVertexMax = vertexMax;
    for (int cc = 0; cc < cellsVertexNumbers.size();++cc)
        if (cellsVertexNumbers[cc]+1 > newVertexMax)
            newVertexMax = cellsVertexNumbers[cc]+1;
    if(newVertexMax > vertexMax)
        growCellVerticesList(newVertexMax);

    //update cell composition and number of composing vertices
    {
    ArrayHandle<int> cv(cellVertices);
    ArrayHandle<int> cvn(cellVertexNum);
    for (int cc = 0; cc < cellsInvolved.size();++cc)
        {
        int cIdx = cellsInvolved[cc];
        int vCur, vNext;
        int vNeighs = cvn.data[cIdx];
        int insertionPosition = 0;
        for (int vv = 0; vv < vNeighs-1; ++vv)
            {
            vCur = cv.data[cellNeighborIndexer(vv,cIdx)];
            vNext = cv.data[cellNeighborIndexer(vv+1,cIdx)];
            if((vCur==vertexIndex1 && vNext==vertexIndex2) || (vCur==vertexIndex2 && vNext==vertexIndex1) )
                insertionPosition = vv + 1;
            }
        cout << "insert vertex " << newVertexIndex << " at position " << insertionPosition << endl;

        int vIndex = 0;
        vector<int> cellComp(vNeighs+1);
        for (int vv = 0; vv < vNeighs; ++vv)
            {
            if(vv == insertionPosition)
                {
                cellComp[vIndex] = newVertexIndex;
                vIndex +=1;
                }
            cellComp[vIndex] = cv.data[cellNeighborIndexer(vv,cIdx)];
            vIndex += 1;
            }
        cvn.data[cIdx] = cellComp.size();
        for (int vv = 0; vv < cellComp.size();++vv)
            {
            cv.data[cellNeighborIndexer(vv,cIdx)] = cellComp[vv];
            printf("new vertex %i: %i\n", vv, cellComp[vv]);
            }
        }

    }//end arrayhandle scope

    };

/*!
This function "splits" a vertex, which I'll note in this comment as V0, into two vertices (V0 and V0'),
such that the bond between them has angle "theta" and norm "separation".
This function is meant to be quite generic, so the coordination number of the target vertex can be
anything (other than one, of course). Because of this, a choice has to be made about how to partition
the original set of vertex Neighbors of V0 into the new neighbors of V0 and those of V0'. I have made
the "geometric" choice. Let v_{ij} be the vector from i to j. Create a list of v_{0i} for all i
corresponding to original neighbors of V0. Then, if the dot product (v_{00'}.v_{0i})>0 i will be a
neighbor of V0', otherwise it'll be a neighbor of V0.
Cell neighbors of V0 and V0' are computed accordingly; some trickery is played to correctly do this
in arbitrary coordination settings.

As a note, the vertices are moved so that the center of the edge between them is where the original
vertex was.
\param vertexIndex The index of the vertex to split
\param separation The desired length of the bond between the original vertex and the new one
\param theta The desired angle (in the lab frame) from the original vertex and the new one
*/
void vertexModelGenericBase::splitVertex(int vertexIndex, Dscalar separation, Dscalar theta)
    {
    //Let's start with the easy parts: growing the relevant lists and assigning positions
    int newVertexIndex = Nvertices;
    Nvertices += 1;

    //add a new index for the spatial sorters
    ittVertex.push_back(Nvertices-1);
    ttiVertex.push_back(Nvertices-1);
    tagToIdxVertex.push_back(Nvertices-1);
    idxToTagVertex.push_back(Nvertices-1);

    //resize per-vertex-coordination lists
    resizePerCoordinationLists();

    //take care of the per-vertex lists
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    growGPUArray(vertexPositions,1);
    growGPUArray(vertexMasses,1);
    growGPUArray(vertexVelocities,1);

    Dscalar2 newV0Pos,newV1Pos;
    //mass is the same as the dividing vertex. velocities of the new vertex is zero. Positions are set
    {//Array Handle scope
        ArrayHandle<Dscalar> masses(vertexMasses);
        ArrayHandle<Dscalar2> velocities(vertexVelocities);
        ArrayHandle<Dscalar2> positions(vertexPositions);
        masses.data[Nvertices-1] = masses.data[vertexIndex];
        Dscalar2 zero = make_Dscalar2(0.0,0.0);
        velocities.data[Nvertices-1] = zero;
        Dscalar2 edge = make_Dscalar2(cos(theta),sin(theta));
        edge = 0.5*separation*edge;
        newV0Pos = positions.data[vertexIndex]-edge;
        Box->putInBoxReal(newV0Pos);
        newV1Pos = positions.data[vertexIndex]+edge;
        Box->putInBoxReal(newV1Pos);
        positions.data[vertexIndex] = newV0Pos;
        positions.data[Nvertices-1] = newV1Pos;
    }//end array handle scope

    vertexNeighborIndexer = Index2D(vertexCoordinationMaximum,Nvertices);
    Index2D oldVNI = Index2D(vertexCoordinationMaximum,Nvertices-1);

    //vertex neighbors and vertex neighbor num
    vector<int> v0vertexNeighs,v0primeVertexNeighs;
    {//scope for array handles
    Dscalar2 disp;
    Dscalar2 v00p;
    Box->minDist(newV0Pos,newV1Pos,v00p);
    ArrayHandle<int> vn(vertexNeighbors);
    ArrayHandle<int> vnn(vertexNeighborNum);
    ArrayHandle<Dscalar2> positions(vertexPositions);

    //what is the list of vertex neighbors that will be assigned to one of the two resolving vertices?
    int neighs = vnn.data[vertexIndex];
    vector<pair<Dscalar, int> > relativeDistanceList;
    for (int nn = 0; nn < neighs; ++nn)
        {
        int neighbor = vn.data[oldVNI(nn,vertexIndex)];
        Dscalar2 vPos = positions.data[neighbor];
        Box->minDist(newV0Pos,vPos,disp);
        Dscalar norm0 = disp.x*disp.x + disp.y*disp.y;
        Box->minDist(newV1Pos,vPos,disp);
        Dscalar norm1 = disp.x*disp.x + disp.y*disp.y;

        relativeDistanceList.push_back(make_pair(norm1 - norm0,neighbor));
        //Dscalar dotProduct = v00p.x*disp.x+v00p.y*disp.y;
        //if(norm1 > norm0)
        //    v0vertexNeighs.push_back(neighbor);
        //else
        //    v0primeVertexNeighs.push_back(neighbor);
        };
    //make sure each vertex always inherits at least one of the two
    //     neighbors -- no vertices of coordination 1 in this code!!!
    std::sort(relativeDistanceList.begin(),relativeDistanceList.end());
    v0primeVertexNeighs.push_back(relativeDistanceList[0].second);
    for (int ii = 1; ii < relativeDistanceList.size()-1; ++ii)
        {
        if(relativeDistanceList[ii].first >= 0)
            v0vertexNeighs.push_back(relativeDistanceList[ii].second);
        else
            v0primeVertexNeighs.push_back(relativeDistanceList[ii].second);
        };
    v0vertexNeighs.push_back(relativeDistanceList[relativeDistanceList.size()-1].second);

    //don't forget the new and old vertices themselves
    v0vertexNeighs.push_back(newVertexIndex);
    v0primeVertexNeighs.push_back(vertexIndex);
    vnn.data[vertexIndex] = v0vertexNeighs.size();
    vnn.data[newVertexIndex] = v0primeVertexNeighs.size();
    //limit vertex neighbors of V0 to just the given subset
    for (int nn = 0; nn < v0vertexNeighs.size(); ++nn)
        vn.data[vertexNeighborIndexer(nn,vertexIndex)] = v0vertexNeighs[nn];
    //add relevant vertex neighbors for the new vertex. Also rewire the connected vertices
    for (int nn = 0; nn < v0primeVertexNeighs.size(); ++nn)
        {
        vn.data[vertexNeighborIndexer(nn,newVertexIndex)] = v0primeVertexNeighs[nn];
        int otherIdx = v0primeVertexNeighs[nn];
        int otherNeighs =  vnn.data[otherIdx];
        for (int jj = 0; jj < otherNeighs; ++jj)
            if (vn.data[vertexNeighborIndexer(jj,otherIdx)] == vertexIndex)
                vn.data[vertexNeighborIndexer(jj,otherIdx)] = newVertexIndex;
        }
    }//end scope


    //vertex cell neighbors
    //vertex cell neighbor num
    //

    //does the maximum number of vertices around a cell need to be incremented?
    //cellNeighborIndexer = Index2D(vertexMax,Ncells);
    //Index2D oldCNI = Index2D(vertexMaxOld????,Ncells);
    //cellVertices
    //cellVertexNum

    }

/*!
cell death, as opposed to cell removal, removes a cell but maintains the confluent nature of the
tissue. This is accomplished by subsequent calls to removeCell and then mergeVertices.
*/
void vertexModelGenericBase::cellDeath(int cellIndex)
    {
    //get a list of vertices that make up the cell
    vector<int> vertices;
    vertices.reserve(vertexMax);
    {
    ArrayHandle<int> cv(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> cvn(cellVertexNum,access_location::host,access_mode::read);
    int neigh = cvn.data[cellIndex];
    for (int nn = 0; nn < neigh; ++nn)
        vertices.push_back(cv.data[cellNeighborIndexer(nn,cellIndex)]);
    };

    //call removeCell
    vector<int> cellsToRemove(1,cellIndex);
    removeCells(cellsToRemove);

    //use vertexMap and vertexMapInverse to figure out the new vertex labels
    vector<int> verticesToMerge;
    verticesToMerge.reserve(vertexCoordinationMaximum);
    for (int vv = 0; vv < vertices.size(); ++vv)
        if (vertexMap[vertices[vv]] != -1)
            verticesToMerge.push_back(vertices[vv]);
    //merge those vertices together
    mergeVertices(verticesToMerge);
    };

/*!
uses an up-to-date cellMap and cellMapInverse to relabel the cell sorting arrays.
Requires that Ncells is the new, not old, value
*/
void vertexModelGenericBase::remapCellSorting()
    {
    //get a map from the tags
    vector<int> tagsToRemove,tagMap,tagMapInverse;
    for (int cc = 0; cc < cellMap.size(); ++cc)
        if( cellMap[cc] == -1)
            tagsToRemove.push_back(idxToTag[cc]);
    sort(tagsToRemove.begin(),tagsToRemove.end());
    createIndexMapAndInverse(tagMap,tagMapInverse,tagsToRemove,cellMap.size());

    vector<int> newitt, newtti, newidxtotag, newtagtoidx;
    newitt.resize(Ncells);newtti.resize(Ncells);newidxtotag.resize(Ncells);newtagtoidx.resize(Ncells);
    for (int cc = 0; cc < Ncells; ++cc)
        {
        int newCellIndex = cellMapInverse[cc];
        int newTagIndex = idxToTag[newCellIndex];
        newidxtotag[cc]=tagMap[newTagIndex];
        //newitt[cc] = tagMap[itt[newCellIndex]];
        };
    for (int cc = 0; cc < Ncells;++cc)
        {
        //newtti[newitt[cc]]=cc;
        newtagtoidx[newidxtotag[cc]]=cc;
        };
    itt=newitt;
    tti=newtti;
    idxToTag=newidxtotag;
    tagToIdx=newtagtoidx;
    };

/*!
uses an up-to-date vertexMap and vertexMapInverse to relabel the vertex sorting arrays.
Requires that Nvertices is the new, not old, value
*/
void vertexModelGenericBase::remapVertexSorting()
    {
    //get a map from the tags
    vector<int> tagsToRemove,tagMap,tagMapInverse;
    for (int vv = 0; vv < vertexMap.size(); ++vv)
        if( vertexMap[vv] == -1)
            tagsToRemove.push_back(idxToTagVertex[vv]);
    sort(tagsToRemove.begin(),tagsToRemove.end());
    createIndexMapAndInverse(tagMap,tagMapInverse,tagsToRemove,vertexMap.size());
    vector<int> newittv, newttiv, newidxtotagv, newtagtoidxv;
    newittv.resize(Nvertices);newttiv.resize(Nvertices);newidxtotagv.resize(Nvertices);newtagtoidxv.resize(Nvertices);
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int newVertexIndex = vertexMapInverse[vv];
        int newTagIndex = idxToTagVertex[newVertexIndex];
        newidxtotagv[vv] = tagMap[newTagIndex];
        //newittv[vv] = tagMap[ittVertex[newVertexIndex]];
        };
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        //newttiv[newittv[vv]]=vv;
        newtagtoidxv[newidxtotagv[vv]]=vv;
        };
    ittVertex=newittv;
    ttiVertex=newttiv;
    idxToTagVertex=newidxtotagv;
    tagToIdxVertex=newtagtoidxv;
    };
