#define ENABLE_CUDA

#include "vertexModelBase.h"
#include "vertexModelBase.cuh"
#include "voronoi2d.h"
/*! \file vertexModelBase.cpp */

/*!
move vertices according to an inpute GPUarray
*/
void vertexModelBase::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements)
    {
    //handle things either on the GPU or CPU
    if (GPUcompute)
        {
        ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
        gpu_vm_displace(d_v.data,
                         d_disp.data,
                         Box,
                         Nvertices);
        }
    else
        {
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
        for (int i = 0; i < Nvertices; ++i)
            {
            h_v.data[i].x += h_disp.data[i].x;
            h_v.data[i].y += h_disp.data[i].y;
            Box.putInBoxReal(h_v.data[i]);
            };
        };
    };

/*!
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures
(the vertices)...compute the area and perimeter of the cells
*/
void vertexModelBase::computeGeometryCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);

    //compute the geometry for each cell
    for (int i = 0; i < Ncells; ++i)
        {
        int neighs = h_nn.data[i];
//      Define the vertices of a cell relative to some (any) of its verties to take care of periodic boundaries
        Dscalar2 cellPos = h_v.data[h_n.data[n_idx(neighs-2,i)]];
        Dscalar2 vlast, vcur,vnext;
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        //compute the vertex position relative to the cell position
        vlast.x=0.;vlast.y=0.0;
        int vidx = h_n.data[n_idx(neighs-1,i)];
        Box.minDist(h_v.data[vidx],cellPos,vcur);
        for (int nn = 0; nn < neighs; ++nn)
            {
            //for easy force calculation, save the current, last, and next vertex position in the approprate spot.
            int forceSetIdx= -1;
            for (int ff = 0; ff < 3; ++ff)
                if(h_vcn.data[3*vidx+ff]==i)
                    forceSetIdx = 3*vidx+ff;

            vidx = h_n.data[n_idx(nn,i)];
            Box.minDist(h_v.data[vidx],cellPos,vnext);

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
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures (the vertices)
*/
void vertexModelBase::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,      access_location::device,access_mode::read);
    ArrayHandle<int>      d_cvn(cellVertexNum,       access_location::device,access_mode::read);
    ArrayHandle<int>      d_cv(cellVertices,         access_location::device,access_mode::read);
    ArrayHandle<int>      d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,             access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,       access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,            access_location::device,access_mode::overwrite);

    gpu_vm_geometry(
                    d_v.data,
                    d_cvn.data,
                    d_cv.data,
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    Ncells,n_idx,Box);
    };

/*!
One would prefer the cell position to be defined as the centroid, requiring an additional computation of the cell area.
This may be implemented some day, but for now we define the cell position as the straight average of the vertex positions.
This isn't really used much, anyway, so update this only when the functionality becomes needed
*/
void vertexModelBase::getCellPositionsCPU()
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);

    Dscalar2 vertex,baseVertex,pos;
    for (int cell = 0; cell < Ncells; ++cell)
        {
        baseVertex = h_v.data[h_n.data[n_idx(0,cell)]];
        int neighs = h_nn.data[cell];
        pos.x=0.0;pos.y=0.0;
        //compute the vertex position relative to the cell position
        for (int n = 1; n < neighs; ++n)
            {
            int vidx = h_n.data[n_idx(n,cell)];
            Box.minDist(h_v.data[vidx],baseVertex,vertex);
            pos.x += vertex.x;
            pos.y += vertex.y;
            };
        pos.x /= neighs;
        pos.y /= neighs;
        pos.x += baseVertex.x;
        pos.y += baseVertex.y;
        Box.putInBoxReal(pos);
        h_p.data[cell] = pos;
        };
    };

/*!
Repeat the above calculation of "cell positions", but on the GPU
*/
void vertexModelBase::getCellPositionsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::read);

    gpu_vm_get_cell_positions(d_p.data,
                               d_v.data,
                               d_cvn.data,
                               d_cv.data,
                               Ncells,
                               n_idx,
                               Box);
    };

/*!
 Initialize the auxilliary edge flip data structures to zero
 */
void vertexModelBase::initializeEdgeFlipLists()
    {
    vertexEdgeFlips.resize(3*Nvertices);
    vertexEdgeFlipsCurrent.resize(3*Nvertices);
    ArrayHandle<int> h_vflip(vertexEdgeFlips,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vflipc(vertexEdgeFlipsCurrent,access_location::host,access_mode::overwrite);
    for (int i = 0; i < 3*Nvertices; ++i)
        {
        h_vflip.data[i]=0;
        h_vflipc.data[i]=0;
        }

    finishedFlippingEdges.resize(1);
    ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::overwrite);
    h_ffe.data[0]=0;
    };

/*!
when a transition increases the maximum number of vertices around any cell in the system,
call this function first to copy over the cellVertices structure into a larger array
 */
void vertexModelBase::growCellVerticesList(int newVertexMax)
    {
    cout << "maximum number of vertices per cell grew from " <<vertexMax << " to " << newVertexMax << endl;
    vertexMax = newVertexMax+1;
    Index2D old_idx = n_idx;
    n_idx = Index2D(vertexMax,Ncells);

    GPUArray<int> newCellVertices;
    newCellVertices.resize(vertexMax*Ncells);
    {//scope for array handles
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n_old(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(newCellVertices,access_location::host,access_mode::readwrite);

    for(int cell = 0; cell < Ncells; ++cell)
        {
        int neighs = h_nn.data[cell];
        for (int n = 0; n < neighs; ++n)
            {
            h_n.data[n_idx(n,cell)] = h_n_old.data[old_idx(n,cell)];
            };
        };
    };//scope for array handles
    cellVertices.resize(vertexMax*Ncells);
    cellVertices.swap(newCellVertices);
    };


/*!
Trigger a cell division event, which involves some laborious re-indexing of various data structures.
This simple version of cell division will take a cell and two specified vertices. The edges emanating
clockwise from each of the two vertices will gain a new vertex in the middle of those edges. A new cell is formed by connecting those two new vertices together.
The vector of "parameters" here should be three integers:
parameters[0] = the index of the cell to undergo a division event
parameters[1] = the first vertex to gain a new (clockwise) vertex neighbor.
parameters[2] = the second .....
The two vertex numbers should be between 0 and celLVertexNum[parameters[0]], respectively, NOT the
indices of the vertices being targeted
\post This function is meant to be called before the start of a new timestep. It should be immediately followed by a computeGeometry call
*/
void vertexModelBase::cellDivision(vector<int> &parameters)
    {
    int cellIdx = parameters[0];
    if(cellIdx >= Ncells)
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    int v1 = min(parameters[1],parameters[2]);
    int v2 = max(parameters[1],parameters[2]);

    Dscalar2 cellPos;
    Dscalar2 newV1Pos,newV2Pos;
    int v1idx, v2idx, v1NextIdx, v2NextIdx;
    int newV1CellNeighbor, newV2CellNeighbor;
    bool increaseVertexMax = false;
    int neighs;
    vector<int> combinedVertices;
    {//scope for array handles
    ArrayHandle<Dscalar2> vP(vertexPositions);
    ArrayHandle<int> cellVertNum(cellVertexNum);
    ArrayHandle<int> cv(cellVertices);
    ArrayHandle<int> vcn(vertexCellNeighbors);
    neighs = cellVertNum.data[cellIdx];

    combinedVertices.reserve(neighs+2);
    for (int i = 0; i < neighs; ++i)
        combinedVertices.push_back(cv.data[n_idx(i,cellIdx)]);
    combinedVertices.insert(combinedVertices.begin()+1+v1,Nvertices);
    combinedVertices.insert(combinedVertices.begin()+2+v2,Nvertices+1);

    if(v1 >= neighs || v2 >=neighs)
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    v1idx = cv.data[n_idx(v1,cellIdx)];
    v2idx = cv.data[n_idx(v2,cellIdx)];
    if (v1 < neighs - 1)
        v1NextIdx = cv.data[n_idx(v1+1,cellIdx)];
    else
        v1NextIdx = cv.data[n_idx(0,cellIdx)];
    if (v2 < neighs - 1)
        v2NextIdx = cv.data[n_idx(v2+1,cellIdx)];
    else
        v2NextIdx = cv.data[n_idx(0,cellIdx)];

    //find the positions of the new vertices
    Dscalar2 disp;
    Box.minDist(vP.data[v1NextIdx],vP.data[v1idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV1Pos = vP.data[v1idx] + disp;
    Box.putInBoxReal(newV1Pos);
    Box.minDist(vP.data[v2NextIdx],vP.data[v2idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV2Pos = vP.data[v2idx] + disp;
    Box.putInBoxReal(newV2Pos);

    //find the third cell neighbor of the new vertices
    int ans = -1;
    for (int vi = 3*v1idx; vi < 3*v1idx+3; ++vi)
        for (int vj = 3*v1NextIdx; vj < 3*v1NextIdx+3; ++vj)
            {
            int c1 = vcn.data[vi];
            int c2 = vcn.data[vj];
            if ((c1 == c2) &&(c1 != cellIdx))
                ans = c1;
            };
    if (ans >=0)
        newV1CellNeighbor = ans;
    else
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    ans = -1;
    for (int vi = 3*v2idx; vi < 3*v2idx+3; ++vi)
        for (int vj = 3*v2NextIdx; vj < 3*v2NextIdx+3; ++vj)
            {
            int c1 = vcn.data[vi];
            int c2 = vcn.data[vj];
            if ((c1 == c2) &&(c1 != cellIdx))
                ans = c1;
            };
    if (ans >=0)
        newV2CellNeighbor = ans;
    else
        {
        printf("\nError in cell division. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    if(cellVertNum.data[newV1CellNeighbor] + 1 >=vertexMax)
        increaseVertexMax = true;
    if(cellVertNum.data[newV2CellNeighbor] + 1 >=vertexMax)
        increaseVertexMax = true;
    }//end scope of old array handles... new vertices and cells identified

    //update cell and vertex number; have access to both new and old indexer if vertexMax changes
    Index2D n_idxOld(vertexMax,Ncells);
    if (increaseVertexMax)
        {
        printf("vertexMax has increased due to cell division\n");
        vertexMax += 2;
        };
    Ncells += 1;
    Nvertices += 2;
    n_idx = Index2D(vertexMax,Ncells);

    //additions to the spatial sorting vectors...
    itt.push_back(Ncells-1);
    tti.push_back(Ncells-1);
    tagToIdx.push_back(Ncells-1);
    idxToTag.push_back(Ncells-1);
    ittVertex.push_back(Nvertices-2); ittVertex.push_back(Nvertices-1);
    ttiVertex.push_back(Nvertices-2); ttiVertex.push_back(Nvertices-1);
    tagToIdxVertex.push_back(Nvertices-2); tagToIdxVertex.push_back(Nvertices-1);
    idxToTagVertex.push_back(Nvertices-2); idxToTagVertex.push_back(Nvertices-1);

    //GPUArrays that just need their length changed
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    initializeEdgeFlipLists(); //function call takes care of EdgeFlips and EdgeFlipsCurrent
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);
    AreaPeri.resize(Ncells);

    //use the copy and grow mechanism where we need to actually set values
    growGPUArray(vertexPositions,2); //(nv)
    growGPUArray(vertexNeighbors,6); //(3*nv)
    growGPUArray(vertexCellNeighbors,6); //(3*nv)
    growGPUArray(cellVertexNum,1); //(nc)
    growGPUArray(AreaPeriPreferences,1); //(nc)
    growGPUArray(Motility,1); //(nc)
    growGPUArray(cellDirectors,1);
    //ADD MODULI if they are implemented in Simple2DCell.h
    //the index cellVertices array needs more care...
    vector<int>  cellVerticesVec;
    copyGPUArrayData(cellVertices,cellVerticesVec);
    cellVertices.resize(vertexMax*Ncells);
    //first, let's take care of the trivial things
        {//arrayhandle scope
        ArrayHandle<Dscalar2> h_mot(Motility); h_mot.data[Ncells-1] = h_mot.data[cellIdx];
        ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences); h_APP.data[Ncells-1] = h_APP.data[cellIdx];
        noiseSource noise;
        noise.Reproducible = Reproducible;
        ArrayHandle<Dscalar> h_cd(cellDirectors); h_cd.data[Ncells-1] = noise.getRealUniform(0.,2*PI);
        ArrayHandle<Dscalar2> h_vp(vertexPositions);
        h_vp.data[Nvertices-2] = newV1Pos;
        h_vp.data[Nvertices-1] = newV2Pos;
        }

    //the vertex-vertex neighbors
        {//arrayHandle scope
        ArrayHandle<int> h_vv(vertexNeighbors);
        //new v1
        h_vv.data[3*(Nvertices-2)+0] = v1idx;
        h_vv.data[3*(Nvertices-2)+1] = v1NextIdx;
        h_vv.data[3*(Nvertices-2)+2] = Nvertices-1;
        //new v2
        h_vv.data[3*(Nvertices-1)+0] = Nvertices-2;
        h_vv.data[3*(Nvertices-1)+1] = v2idx;
        h_vv.data[3*(Nvertices-1)+2] = v2NextIdx;
        //v1idx
        for (int ii = 3*v1idx; ii < 3*(v1idx+1); ++ii)
            if (h_vv.data[ii] == v1NextIdx) h_vv.data[ii] = Nvertices-2;
        //v1NextIdx
        for (int ii = 3*v1NextIdx; ii < 3*(v1NextIdx+1); ++ii)
            if (h_vv.data[ii] == v1idx) h_vv.data[ii] = Nvertices-2;
        //v2idx
        for (int ii = 3*v2idx; ii < 3*(v2idx+1); ++ii)
            if (h_vv.data[ii] == v2NextIdx) h_vv.data[ii] = Nvertices-1;
        //v2NextIdx
        for (int ii = 3*v2NextIdx; ii < 3*(v2NextIdx+1); ++ii)
            if (h_vv.data[ii] == v2idx) h_vv.data[ii] = Nvertices-1;
        };

    //for computing vertex-cell neighbors and cellVertices, recall that combinedVertices is a list:
    //v0, v1.. newvertex 1... newvertex2 ... v_old_last_vertex
    //for convenience, rotate this so that it is newvertex1 ... newvertex2, (other vertices), and
    //create another vector that is newvertex2...newvertex1, (other vertices)
    vector<int> cv2=combinedVertices;
    rotate(cv2.begin(), cv2.begin()+v2+2, cv2.end());
    rotate(combinedVertices.begin(), combinedVertices.begin()+v1+1, combinedVertices.end());
    int nVertNewCell = (v2 - v1) +2;
    int nVertCellI = neighs+2-(v2-v1);
        {//arrayHandle scope
        ArrayHandle<int> h_cvn(cellVertexNum);
        h_cvn.data[Ncells-1] = nVertNewCell;
        h_cvn.data[cellIdx] = nVertCellI;

        ArrayHandle<int> h_vcn(vertexCellNeighbors);
        //new v1
        h_vcn.data[3*(Nvertices-2)+0] = newV1CellNeighbor;
        h_vcn.data[3*(Nvertices-2)+1] = Ncells-1;
        h_vcn.data[3*(Nvertices-2)+2] = cellIdx;
        //new v2
        h_vcn.data[3*(Nvertices-1)+0] = newV2CellNeighbor;
        h_vcn.data[3*(Nvertices-1)+1] = cellIdx;
        h_vcn.data[3*(Nvertices-1)+2] = Ncells-1;
        //vertices in between newV1 and newV2 don't neighbor the divided cell any more
        for (int i = 1; i < nVertNewCell-1; ++i)
            for (int vv = 0; vv < 3; ++vv)
                if(h_vcn.data[3*combinedVertices[i]+vv] == cellIdx)
                    h_vcn.data[3*combinedVertices[i]+vv] = Ncells-1;
        };

    // finally, reset the vertices associated with every cell
        {//arrayHandle scope
        ArrayHandle<int> cv(cellVertices);
        ArrayHandle<int> h_cvn(cellVertexNum);
        //first, copy over the old cells with any new indexing
        for (int cell = 0; cell < Ncells -1; ++cell)
            {
            int ns = h_cvn.data[cell];
            for (int vv = 0; vv < ns; ++vv)
                cv.data[n_idx(vv,cell)] = cellVerticesVec[n_idxOld(vv,cell)];
            };
        //correct cellIdx's vertices
        for (int vv = 0; vv < nVertCellI; ++vv)
            cv.data[n_idx(vv,cellIdx)] = cv2[vv];
        //add the vertices to the new cell
        for (int vv = 0; vv < nVertNewCell; ++vv)
            cv.data[n_idx(vv,Ncells-1)] = combinedVertices[vv];

        //insert the vertices into newV1CellNeighbor and newV2CellNeighbor
        vector<int> cn1, cn2;
        int cn1Size = h_cvn.data[newV1CellNeighbor];
        int cn2Size = h_cvn.data[newV2CellNeighbor];
        cn1.reserve(cn1Size+1);
        cn2.reserve(cn2Size+1);
        for (int i = 0; i < cn1Size; ++i)
            {
            int curVertex = cv.data[n_idx(i,newV1CellNeighbor)];
            cn1.push_back(curVertex);
            if(curVertex == v1NextIdx)
                cn1.push_back(Nvertices-2);
            };
        for (int i = 0; i < cn2Size; ++i)
            {
            int curVertex = cv.data[n_idx(i,newV2CellNeighbor)];
            cn2.push_back(curVertex);
            if(curVertex == v2NextIdx)
                cn2.push_back(Nvertices-1);
            };

        //correct newV1CellNeighbor's vertices
        for (int vv = 0; vv < cn1Size+1; ++vv)
            cv.data[n_idx(vv,newV1CellNeighbor)] = cn1[vv];
        //correct newV2CellNeighbor's vertices
        for (int vv = 0; vv < cn2Size+1; ++vv)
            cv.data[n_idx(vv,newV2CellNeighbor)] = cn2[vv];
        //correct the number of vertex neighbors of the cells
        h_cvn.data[newV1CellNeighbor] = cn1Size+1;
        h_cvn.data[newV2CellNeighbor] = cn2Size+1;
        };
    };
