#define ENABLE_CUDA

#include "vertexModelBase.h"
#include "vertexModelBase.cuh"
#include "voronoiQuadraticEnergy.h"
/*! \file vertexModelBase.cpp */

/*!
move vertices according to an inpute GPUarray
*/
void vertexModelBase::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements,Dscalar scale)
    {
    forcesUpToDate = false;
    //handle things either on the GPU or CPU
    if (GPUcompute)
        {
        ArrayHandle<Dscalar2> d_d(displacements,access_location::device,access_mode::read);
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
        if (scale == 1.)
            gpu_move_degrees_of_freedom(d_v.data,d_d.data,Nvertices,*(Box));
        else
            gpu_move_degrees_of_freedom(d_v.data,d_d.data,scale,Nvertices,*(Box));
        }
    else
        {
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
        if(scale ==1.)
            {
            for (int i = 0; i < Nvertices; ++i)
                {
                h_v.data[i].x += h_disp.data[i].x;
                h_v.data[i].y += h_disp.data[i].y;
                Box->putInBoxReal(h_v.data[i]);
                };
            }
        else
            {
            for (int i = 0; i < Nvertices; ++i)
                {
                h_v.data[i].x += scale*h_disp.data[i].x;
                h_v.data[i].y += scale*h_disp.data[i].y;
                Box->putInBoxReal(h_v.data[i]);
                };
            }
        };
    };

/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void vertexModelBase::initializeVertexModelBase(int n,bool spvInitialize)
    {
    //set number of cells, and call initializer chain
    Ncells=n;
    initializeSimple2DActiveCell(Ncells);
    //derive the vertices from a voronoi tesselation
    setCellsVoronoiTesselation(spvInitialize);

    setT1Threshold(0.01);
    //initializes per-cell lists
    initializeCellSorting();
    cellEdgeFlips.resize(Ncells);
    vector<int> ncz(Ncells,0);
    fillGPUArrayWithVector(ncz,cellEdgeFlips);

    vertexMasses.resize(Nvertices);
    vertexVelocities.resize(Nvertices);
    vector<Dscalar> vmasses(Nvertices,1.0);
    fillGPUArrayWithVector(vmasses,vertexMasses);
    vector<Dscalar2> velocities(Nvertices,make_Dscalar2(0.0,0.0));
    fillGPUArrayWithVector(velocities,vertexVelocities);

    //initializes per-vertex lists
    displacements.resize(Nvertices);
    initializeVertexSorting();
    initializeEdgeFlipLists();

    //initialize per-triple-vertex lists
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);
    cellSets.resize(3*Nvertices);

    growCellVertexListAssist.resize(1);
    ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::overwrite);
    h_grow.data[0]=0;
    };

/*!
enforce and update topology of vertex wiring on either the GPU or CPU
*/
void vertexModelBase::enforceTopology()
    {
    if(GPUcompute)
        {
        //see if vertex motion leads to T1 transitions...ONLY allow one transition per vertex and
        //per cell per timestep
        testAndPerformT1TransitionsGPU();
        }
    else
        {
        //see if vertex motion leads to T1 transitions
        testAndPerformT1TransitionsCPU();
        };
    };

/*!
A function of convenience.... initialize cell positions and vertices by constructing the Delaunay
triangulation of the current cell positions. If you want something more regular, run the Voronoi mode for a few
timesteps to smooth out the random point set first.
\param spvInitialize only use if the initial cell positions are to be random, and you want to make the points more uniform
\post After this is called, all topology data structures are initialized
*/
void vertexModelBase::setCellsVoronoiTesselation(bool spvInitialize)
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    //use the Voronoi class to relax the initial configuration just a bit?
    if(spvInitialize)
        {
        EOMPtr spp = make_shared<selfPropelledParticleDynamics>(Ncells);

        ForcePtr spv = make_shared<Voronoi2D>(Ncells,1.0,3.8,Reproducible);
        spv->setCellPreferencesUniform(1.0,3.8);
        spv->setv0Dr(.1,1.0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(spv);
        sim->addUpdater(spp,spv);
        sim->setIntegrationTimestep(0.1);
        sim->setCPUOperation(true);
        spp->setDeltaT(0.1);
        sim->setReproducible(true);

        for (int ii = 0; ii < 100;++ii)
            sim->performTimestep();
        ArrayHandle<Dscalar2> h_pp(spv->cellPositions,access_location::host,access_mode::read);
        for (int ii = 0; ii < Ncells; ++ii)
            h_p.data[ii] = h_pp.data[ii];
        };

    //call CGAL to get Delaunay triangulation
    vector<pair<Point,int> > Psnew(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Psnew[ii]=make_pair(Point(h_p.data[ii].x,h_p.data[ii].y),ii);
        };
    Dscalar b11,b12,b21,b22;
    Box->getBoxDims(b11,b12,b21,b22);
    Iso_rectangle domain(0.0,0.0,b11,b22);
    PDT T(Psnew.begin(),Psnew.end(),domain);
    T.convert_to_1_sheeted_covering();

    //set number of vertices
    Nvertices = 2*Ncells;
    vertexPositions.resize(Nvertices);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::overwrite);

    //first, ask CGAL for the circumcenter of the face, and add it to the list of vertices, and make a map between the iterator and the vertex idx
    map<PDT::Face_handle,int> faceToVoroIdx;
    int idx = 0;
    for(PDT::Face_iterator fit = T.faces_begin(); fit != T.faces_end(); ++fit)
        {
        PDT::Point p(T.dual(fit));
        h_v.data[idx].x = p.x();
        h_v.data[idx].y = p.y();
        faceToVoroIdx[fit] = idx;
        idx +=1;
        };
    //create a list of what vertices are connected to what vertices,
    //and what cells each vertex is part of
    vertexNeighbors.resize(3*Nvertices);
    vertexCellNeighbors.resize(3*Nvertices);
    ArrayHandle<int> h_vn(vertexNeighbors,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::overwrite);
    for(PDT::Face_iterator fit = T.faces_begin(); fit != T.faces_end(); ++fit)
        {
        int vidx = faceToVoroIdx[fit];
        for(int ff =0; ff<3; ++ff)
            {
            PDT::Face_handle neighFace = fit->neighbor(ff);
            int vnidx = faceToVoroIdx[neighFace];
            h_vn.data[3*vidx+ff] = vnidx;
            h_vcn.data[3*vidx+ff] = fit->vertex(ff)->info();
            };
        };

    //now create a list of what vertices are associated with each cell
    //first get the maximum number of vertices for a cell, and the number of vertices per cell
    cellVertexNum.resize(Ncells);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::overwrite);
    vertexMax = 0;
    int nnum = 0;
    for(PDT::Vertex_iterator vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
        {
        PDT::Vertex_circulator vc(vit);
        int base = vc ->info();
        int neighs = 1;
        ++vc;
        while(vc->info() != base)
            {
            neighs += 1;
            ++vc;
            };
        h_cvn.data[vit->info()] = neighs;
        if (neighs > vertexMax) vertexMax = neighs;
        nnum += neighs;
        };
    vertexMax += 4;
    vertexMax = 30;
    cout << "Total number of neighs = " << nnum << endl;
    cellVertices.resize(vertexMax*Ncells);
    n_idx = Index2D(vertexMax,Ncells);

    //now use face circulators and the map to get the vertices associated with each cell
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::overwrite);
    for(PDT::Vertex_iterator vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
        {
        int cellIdx = vit->info();
        PDT::Face_circulator fc(vit);
        int fidx = 0;
        for (int ff = 0; ff < h_cvn.data[vit->info()]; ++ff)
            {
            h_cv.data[n_idx(fidx,cellIdx)] = faceToVoroIdx[fc];
            ++fidx;
            ++fc;
            };
        };
   };

/*!
 *When sortPeriod < 0 this routine does not get called
 \post vertices are re-ordered according to a Hilbert sorting scheme, cells are reordered according
 to what vertices they are near, and all data structures are updated
 */
void vertexModelBase::spatialSorting()
    {
    //the base vertex model class doesn't need to change any other unusual data structures at the moment
    spatiallySortVerticesAndCellActivity();
    reIndexVertexArray(vertexMasses);
    reIndexVertexArray(vertexVelocities);
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
        Box->minDist(h_v.data[vidx],cellPos,vcur);
        for (int nn = 0; nn < neighs; ++nn)
            {
            //for easy force calculation, save the current, last, and next vertex position in the approprate spot.
            int forceSetIdx= -1;
            for (int ff = 0; ff < 3; ++ff)
                if(h_vcn.data[3*vidx+ff]==i)
                    forceSetIdx = 3*vidx+ff;
            vidx = h_n.data[n_idx(nn,i)];
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
                    Ncells,n_idx,*(Box));
    };

/*!
This function fills the "cellPositions" GPUArray with the centroid of every cell. Does not assume
that the area in the AreaPeri array is current. This function just calls the CPU or GPU routine, as determined by the GPUcompute flag
*/
void vertexModelBase::getCellCentroids()
    {
    if(GPUcompute)
        getCellCentroidsGPU();
    else
        getCellCentroidsCPU();
    };

/*!
GPU computation of the centroid of every cell
*/
void vertexModelBase::getCellCentroidsGPU()
    {
    printf("getCellCentroidsGPU() function not currently functional...Very sorry\n");
    throw std::exception();
    };

/*!
CPU computation of the centroid of every cell
*/
void vertexModelBase::getCellCentroidsCPU()
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);

    Dscalar2 zero = make_Dscalar2(0.0,0.0);
    Dscalar2 baseVertex;
    for (int cell = 0; cell < Ncells; ++cell)
        {
        //for convenience, for each cell we will make a vector of the vertices of the cell relative to vertex 0
        //the vector will be of length (vertices+1), and the first and last entry will be zero.
        baseVertex = h_v.data[h_n.data[n_idx(0,cell)]];
        int neighs = h_nn.data[cell];
        vector<Dscalar2> vertices(neighs+1,zero);
        for (int vv = 1; vv < neighs; ++vv)
            {
            int vidx = h_n.data[n_idx(vv,cell)];
            Box->minDist(h_v.data[vidx],baseVertex,vertices[vv]);
            };
        //compute the area and the sums for the centroids
        Dscalar Area = 0.0;
        Dscalar2 centroid = zero;
        for (int vv = 0; vv < neighs; ++vv)
            {
            Area += (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            centroid.x += (vertices[vv].x+vertices[vv+1].x) * (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            centroid.y += (vertices[vv].y+vertices[vv+1].y) * (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            };
        Area = 0.5*Area;
        centroid.x = centroid.x / (6.0*Area) + baseVertex.x;
        centroid.y = centroid.y / (6.0*Area) + baseVertex.y;
        Box->putInBoxReal(centroid);
        h_p.data[cell] = centroid;
        };
    };

/*!
This function fills the "cellPositions" GPUArray with the mean position of the vertices of each cell.
This function just calls the CPU or GPU routine, as determined by the GPUcompute flag
*/
void vertexModelBase::getCellPositions()
    {
    if(GPUcompute)
        getCellPositionsGPU();
    else
        getCellPositionsCPU();
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
            Box->minDist(h_v.data[vidx],baseVertex,vertex);
            pos.x += vertex.x;
            pos.y += vertex.y;
            };
        pos.x /= neighs;
        pos.y /= neighs;
        pos.x += baseVertex.x;
        pos.y += baseVertex.y;
        Box->putInBoxReal(pos);
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
                               *(Box));
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

    finishedFlippingEdges.resize(2);
    ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::overwrite);
    h_ffe.data[0]=0;
    h_ffe.data[1]=0;
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
A utility function for the CPU T1 transition routine. Given two vertex indices representing an edge that will undergo
a T1 transition, return in the pass-by-reference variables a helpful representation of the cells in the T1
and the vertices to be re-wired...see the comments in "testAndPerformT1TransitionsCPU" for what that representation is
*/
void vertexModelBase::getCellVertexSetForT1(int vertex1, int vertex2, int4 &cellSet, int4 &vertexSet, bool &growList)
    {
    int cell1,cell2,cell3,ctest;
    int vlast, vcur, vnext, cneigh;
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    cell1 = h_vcn.data[3*vertex1];
    cell2 = h_vcn.data[3*vertex1+1];
    cell3 = h_vcn.data[3*vertex1+2];
    //cell_l doesn't contain vertex 1, so it is the cell neighbor of vertex 2 we haven't found yet
    for (int ff = 0; ff < 3; ++ff)
        {
        ctest = h_vcn.data[3*vertex2+ff];
        if(ctest != cell1 && ctest != cell2 && ctest != cell3)
            cellSet.w=ctest;
        };
    //find vertices "c" and "d"
    cneigh = h_cvn.data[cellSet.w];
    vlast = h_cv.data[ n_idx(cneigh-2,cellSet.w) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cell1)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };

    //classify cell1
    cneigh = h_cvn.data[cell1];
    vlast = h_cv.data[ n_idx(cneigh-2,cell1) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cell1) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cell1)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell1;
    else if(vnext == vertex2)
        cellSet.z = cell1;
    else
        {
        cellSet.y = cell1;
        };

    //classify cell2
    cneigh = h_cvn.data[cell2];
    vlast = h_cv.data[ n_idx(cneigh-2,cell2) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cell2) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cell2)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell2;
    else if(vnext == vertex2)
        cellSet.z = cell2;
    else
        {
        cellSet.y = cell2;
        };

    //classify cell3
    cneigh = h_cvn.data[cell3];
    vlast = h_cv.data[ n_idx(cneigh-2,cell3) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cell3) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cell3)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell3;
    else if(vnext == vertex2)
        cellSet.z = cell3;
    else
        {
        cellSet.y = cell3;
        };

    //get the vertexSet by examining cells j and l
    cneigh = h_cvn.data[cellSet.y];
    vlast = h_cv.data[ n_idx(cneigh-2,cellSet.y) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cellSet.y) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cellSet.y)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.x=vlast;
    vertexSet.y=vnext;
    cneigh = h_cvn.data[cellSet.w];
    vlast = h_cv.data[ n_idx(cneigh-2,cellSet.w) ];
    vcur = h_cv.data[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = h_cv.data[n_idx(cn,cellSet.w)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.w=vlast;
    vertexSet.z=vnext;

    //Does the cell-vertex-neighbor data structure need to be bigger...for safety check all cell-vertex numbers, even if it won't be incremented?
    if(h_cvn.data[cellSet.x] == vertexMax || h_cvn.data[cellSet.y] == vertexMax || h_cvn.data[cellSet.z] == vertexMax || h_cvn.data[cellSet.w] == vertexMax)
        growList = true;
    };

/*!
Test whether a T1 needs to be performed on any edge by simply checking if the edge length is beneath a threshold.
This function also performs the transition and maintains the auxiliary data structures
 */
void vertexModelBase::testAndPerformT1TransitionsCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vn(vertexNeighbors,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_cv(cellVertices,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::readwrite);

    Dscalar2 edge;
    //first, scan through the list for any T1 transitions...
    int vertex2;
    //keep track of whether vertexMax needs to be increased
    int vMax = vertexMax;
    /*
     The following is the convention:
     cell i: contains both vertex 1 and vertex 2, in CW order
     cell j: contains only vertex 1
     cell k: contains both vertex 1 and vertex 2, in CCW order
     cell l: contains only vertex 2
     */
    int4 cellSet;
    /*
    vertexSet (a,b,c,d) have those indices in which before the transition
    cell i has CCW vertices: ..., c, v2, v1, a, ...
    and
    cell k has CCW vertices: ..., b,v1,v2,d, ...
    */
    int4 vertexSet;
    Dscalar2 v1,v2;
    for (int vertex1 = 0; vertex1 < Nvertices; ++vertex1)
        {
        v1 = h_v.data[vertex1];
        //look at vertexNeighbors list for neighbors of vertex, compute edge length
        for (int vv = 0; vv < 3; ++vv)
            {
            vertex2 = h_vn.data[3*vertex1+vv];
            //only look at each pair once
            if(vertex1 < vertex2)
                {
                v2 = h_v.data[vertex2];
                Box->minDist(v1,v2,edge);
                if(norm(edge) < T1Threshold)
                    {
                    bool growCellVertexList = false;
                    getCellVertexSetForT1(vertex1,vertex2,cellSet,vertexSet,growCellVertexList);
                    //forbid a T1 transition that would shrink a triangular cell
                    if( h_cvn.data[cellSet.x] == 3 || h_cvn.data[cellSet.z] == 3)
                        continue;
                    //Does the cell-vertex-neighbor data structure need to be bigger?
                    if(growCellVertexList)
                        {
                        vMax +=1;
                        growCellVerticesList(vMax);
                        h_cv = ArrayHandle<int>(cellVertices,access_location::host,access_mode::readwrite);
                        };

                    //Rotate the vertices in the edge and set them at twice their original distance
                    Dscalar2 midpoint;
                    midpoint.x = v2.x + 0.5*edge.x;
                    midpoint.y = v2.y + 0.5*edge.y;

                    v1.x = midpoint.x-edge.y;
                    v1.y = midpoint.y+edge.x;
                    v2.x = midpoint.x+edge.y;
                    v2.y = midpoint.y-edge.x;
                    Box->putInBoxReal(v1);
                    Box->putInBoxReal(v2);
                    h_v.data[vertex1] = v1;
                    h_v.data[vertex2] = v2;

                    //re-wire the cells and vertices
                    //start with the vertex-vertex and vertex-cell  neighbors
                    for (int vert = 0; vert < 3; ++vert)
                        {
                        //vertex-cell neighbors
                        if(h_vcn.data[3*vertex1+vert] == cellSet.z)
                            h_vcn.data[3*vertex1+vert] = cellSet.w;
                        if(h_vcn.data[3*vertex2+vert] == cellSet.x)
                            h_vcn.data[3*vertex2+vert] = cellSet.y;
                        //vertex-vertex neighbors
                        if(h_vn.data[3*vertexSet.y+vert] == vertex1)
                            h_vn.data[3*vertexSet.y+vert] = vertex2;
                        if(h_vn.data[3*vertexSet.z+vert] == vertex2)
                            h_vn.data[3*vertexSet.z+vert] = vertex1;
                        if(h_vn.data[3*vertex1+vert] == vertexSet.y)
                            h_vn.data[3*vertex1+vert] = vertexSet.z;
                        if(h_vn.data[3*vertex2+vert] == vertexSet.z)
                            h_vn.data[3*vertex2+vert] = vertexSet.y;
                        };
                    //now rewire the cells
                    //cell i loses v2 as a neighbor
                    int cneigh = h_cvn.data[cellSet.x];
                    int cidx = 0;
                    for (int cc = 0; cc < cneigh-1; ++cc)
                        {
                        if(h_cv.data[n_idx(cc,cellSet.x)] == vertex2)
                            cidx +=1;
                        h_cv.data[n_idx(cc,cellSet.x)] = h_cv.data[n_idx(cidx,cellSet.x)];
                        cidx +=1;
                        };
                    h_cvn.data[cellSet.x] -= 1;

                    //cell j gains v2 in between v1 and b
                    cneigh = h_cvn.data[cellSet.y];
                    vector<int> cvcopy1(cneigh+1);
                    cidx = 0;
                    for (int cc = 0; cc < cneigh; ++cc)
                        {
                        int cellIndex = h_cv.data[n_idx(cc,cellSet.y)];
                        cvcopy1[cidx] = cellIndex;
                        cidx +=1;
                        if(cellIndex == vertex1)
                            {
                            cvcopy1[cidx] = vertex2;
                            cidx +=1;
                            };
                        };
                    for (int cc = 0; cc < cneigh+1; ++cc)
                        h_cv.data[n_idx(cc,cellSet.y)] = cvcopy1[cc];
                    h_cvn.data[cellSet.y] += 1;

                    //cell k loses v1 as a neighbor
                    cneigh = h_cvn.data[cellSet.z];
                    cidx = 0;
                    for (int cc = 0; cc < cneigh-1; ++cc)
                        {
                        if(h_cv.data[n_idx(cc,cellSet.z)] == vertex1)
                            cidx +=1;
                        h_cv.data[n_idx(cc,cellSet.z)] = h_cv.data[n_idx(cidx,cellSet.z)];
                        cidx +=1;
                        };
                    h_cvn.data[cellSet.z] -= 1;

                    //cell l gains v1 in between v2 and a
                    cneigh = h_cvn.data[cellSet.w];
                    vector<int> cvcopy2(cneigh+1);
                    cidx = 0;
                    for (int cc = 0; cc < cneigh; ++cc)
                        {
                        int cellIndex = h_cv.data[n_idx(cc,cellSet.w)];
                        cvcopy2[cidx] = cellIndex;
                        cidx +=1;
                        if(cellIndex == vertex2)
                            {
                            cvcopy2[cidx] = vertex1;
                            cidx +=1;
                            };
                        };
                    for (int cc = 0; cc < cneigh+1; ++cc)
                        h_cv.data[n_idx(cc,cellSet.w)] = cvcopy2[cc];
                    h_cvn.data[cellSet.w] = cneigh + 1;

                    };//end condition that a T1 transition should occur
                };
            };//end loop over vertex2
        };//end loop over vertices
    };

/*!
perform whatever check is desired for T1 transtions (here just a "is the edge too short")
and detect whether the edge needs to grow. If so, grow it!
*/
void vertexModelBase::testEdgesForT1GPU()
    {
        {//provide scope for array handles
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
        ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::read);
        ArrayHandle<int> d_vflip(vertexEdgeFlips,access_location::device,access_mode::overwrite);
        ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
        ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::read);
        ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
        ArrayHandle<int> d_grow(growCellVertexListAssist,access_location::device,access_mode::readwrite);

        //first, test every edge, and check if the cellVertices list needs to be grown
        gpu_vm_test_edges_for_T1(d_v.data,
                              d_vn.data,
                              d_vflip.data,
                              d_vcn.data,
                              d_cvn.data,
                              d_cv.data,
                              *(Box),
                              T1Threshold,
                              Nvertices,
                              vertexMax,
                              d_grow.data,
                              n_idx);
        }
    ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::readwrite);
    if(h_grow.data[0] ==1)
        {
        h_grow.data[0]=0;
        growCellVerticesList(vertexMax+1);
        };
    };

/*!
  Iterate through the vertexEdgeFlips list, selecting at most one T1 transition per cell to be done
  on each iteration, until all necessary T1 events have bee performed.
 */
void vertexModelBase::flipEdgesGPU()
    {
    bool keepFlipping = true;
    //By construction, this loop must always run at least twice...save one of the memory transfers
    int iterations = 0;
    while(keepFlipping)
        {
            {//provide scope for ArrayHandles in the multiple-flip-parsing stage
            ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflip(vertexEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflipcur(vertexEdgeFlipsCurrent,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ffe(finishedFlippingEdges,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ef(cellEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int4> d_cs(cellSets,access_location::device,access_mode::readwrite);

            gpu_zero_array(d_ef.data,Ncells);

            gpu_vm_parse_multiple_flips(d_vflip.data,
                               d_vflipcur.data,
                               d_vn.data,
                               d_vcn.data,
                               d_cvn.data,
                               d_cv.data,
                               d_ffe.data,
                               d_ef.data,
                               d_cs.data,
                               n_idx,
                               Ncells);
            };
        //do we need to flip edges? Loop additional times?
        ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::readwrite);
        if(h_ffe.data[0] != 0)
            {
            ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vflipcur(vertexEdgeFlipsCurrent,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ef(cellEdgeFlips,access_location::device,access_mode::readwrite);
            ArrayHandle<int4> d_cs(cellSets,access_location::device,access_mode::readwrite);
            
            gpu_vm_flip_edges(d_vflipcur.data,
                               d_v.data,
                               d_vn.data,
                               d_vcn.data,
                               d_cvn.data,
                               d_cv.data,
                               d_ef.data,
                               d_cs.data,
                               *(Box),
                               n_idx,
                               Nvertices,
                               Ncells);
            iterations += 1;
            };
        if(h_ffe.data[1]==0)
            keepFlipping = false;

        h_ffe.data[0]=0;
        h_ffe.data[1]=0;
        };//end while loop
    };

/*!
Because the cellVertexList might need to grow, it's convenient to break this into two parts
*/
void vertexModelBase::testAndPerformT1TransitionsGPU()
    {
    testEdgesForT1GPU();
    flipEdgesGPU();
    };

/*!
Trigger a cell death event. This REQUIRES that the vertex model cell to die be a triangle (i.e., we
are mimicking a T2 transition)
*/
void vertexModelBase::cellDeath(int cellIndex)
    {
    //first, throw an error if function is called inappropriately
        {
    ArrayHandle<int> h_cvn(cellVertexNum);
    if (h_cvn.data[cellIndex] != 3)
        {
        printf("Error in vertexModelBase::cellDeath... you are trying to perfrom a T2 transition on a cell which is not a triangle\n");
        throw std::exception();
        };
        }
    //Our strategy will be to completely re-wire everything, and then get rid of the dead entries
    //get the cell and vertex identities of the triangular cell and the cell neighbors
    vector<int> cells(3);
    //For conveniences, we will rotate the elements of "vertices" so that the smallest integer is first
    vector<int> vertices(3);
    //also get the vertex neighbors of the vertices (that aren't already part of "vertices")
    vector<int> newVertexNeighbors;
    //So, first create a scope for array handles to write in the re-wired connections
        {//scope for array handle
    ArrayHandle<int> h_cv(cellVertices);
    ArrayHandle<int> h_cvn(cellVertexNum);
    ArrayHandle<int> h_vcn(vertexCellNeighbors);
    int cellsNum=0;
    int smallestV = Nvertices + 1;
    int smallestVIndex = 0;
    for (int vv = 0; vv < 3; ++vv)
        {
        int vIndex = h_cv.data[n_idx(vv,cellIndex)];
        vertices[vv] = vIndex;
        if(vIndex < smallestV)
            {
            smallestV = vIndex;
            smallestVIndex = vv;
            };
        for (int cc =0; cc <3; ++cc)
            {
            int newCell = h_vcn.data[3*vertices[vv]+cc];
            if (newCell == cellIndex) continue;
            bool alreadyFound = false;
            if(cellsNum > 0)
                for (int c2 = 0; c2 < cellsNum; ++c2)
                    if (newCell == cells[c2]) alreadyFound = true;
            if (!alreadyFound)
                {
                cells[cellsNum] = newCell;
                cellsNum +=1;
                }
            };
        };
    std::rotate(vertices.begin(),vertices.begin()+smallestVIndex,vertices.end());
    ArrayHandle<int> h_vn(vertexNeighbors);
    //let's find the vertices connected to the three vertices that form the dying cell
    for (int vv = 0; vv < 3; ++vv)
        {
        for (int v2 = 0; v2 < 3; ++v2)
            {
            int testVertex = h_vn.data[3*vertices[vv]+v2];
            if(testVertex != vertices[0] && testVertex != vertices[1] && testVertex != vertices[2])
                newVertexNeighbors.push_back(testVertex);
            };
        };
    removeDuplicateVectorElements(newVertexNeighbors);
    if(newVertexNeighbors.size() != 3)
        {
        printf("\nError in cell death. File %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        };

    //Eventually, put the new vertex in, say, the centroid... for now, just put it on top of v1
    Dscalar2 newVertexPosition;
    ArrayHandle<Dscalar2> h_v(vertexPositions);
    newVertexPosition = h_v.data[vertices[0]];

    //First, we start updating the data structures
    //new position of the remaining vertex
    h_v.data[vertices[0]] = newVertexPosition;

    //cell vertices and cell vertex number
    for (int oldCell = 0; oldCell < 3; ++oldCell)
        {
        int cIdx = cells[oldCell];
        int neigh = h_cvn.data[cIdx];
        //fun solution: if the cell includes either v2 or v2, replace with v1 and delete duplicates
        vector<int> vNeighs(neigh);
        for (int vv = 0; vv < neigh; ++vv)
            {
            int vIdx = h_cv.data[n_idx(vv,cIdx)];
            if (vIdx == vertices[1] || vIdx==vertices[2])
                vNeighs[vv] = vertices[0];
            else
                vNeighs[vv] = vIdx;
            };
        removeDuplicateVectorElements(vNeighs);
        h_cvn.data[cIdx] = vNeighs.size();
        for (int vv = 0; vv < vNeighs.size(); ++vv)
            h_cv.data[n_idx(vv,cIdx)] = vNeighs[vv];
        };

    //vertex-vertex and vertex-cell neighbors
    for (int ii = 0; ii < 3; ++ii)
        {
        h_vcn.data[3*vertices[0]+ii] = cells[ii];
        h_vcn.data[3*vertices[1]+ii] = cells[ii];
        h_vcn.data[3*vertices[2]+ii] = cells[ii];
        h_vn.data[3*vertices[0]+ii] = newVertexNeighbors[ii];
        for (int vv = 0; vv < 3; ++vv)
            {
            if (h_vn.data[3*newVertexNeighbors[ii]+vv] == vertices[1] ||
                    h_vn.data[3*newVertexNeighbors[ii]+vv] == vertices[2])
                h_vn.data[3*newVertexNeighbors[ii]+vv] = vertices[0];
            };
        };

    //finally (gross), we need to comb through the data arrays and decrement cell indices greater than cellIdx
    //along with vertex numbers greater than v1 and/or v2
    int v1 = std::min(vertices[1],vertices[2]);
    int v2 = std::max(vertices[1],vertices[2]);
    for (int cv = 0; cv < cellVertices.getNumElements(); ++cv)
        {
        int cellVert = h_cv.data[cv];
        if (cellVert >= v1)
            {
            cellVert -= 1;
            if (cellVert >=v2) cellVert -=1;
            h_cv.data[cv] = cellVert;
            }
        };
    for (int vv = 0; vv < vertexNeighbors.getNumElements(); ++vv)
        {
        int vIdx = h_vn.data[vv];
        if (vIdx >= v1)
            {
            vIdx = vIdx - 1;
            if (vIdx >= v2) vIdx = vIdx - 1;
            h_vn.data[vv] = vIdx;
            };
        };
    for (int vv = 0; vv < vertexCellNeighbors.getNumElements(); ++vv)
        {
        int cIdx = h_vcn.data[vv];
        if (cIdx >= cellIndex)
            h_vcn.data[vv] = cIdx - 1;
        };

        };//scope for array handle... now we get to delete choice array elements

    //Now that the GPUArrays have updated data, let's delete elements from the GPUArrays
    vector<int> vpDeletions = {vertices[1],vertices[2]};
    vector<int> vnDeletions = {3*vertices[1],3*vertices[1]+1,3*vertices[1]+2,
                               3*vertices[2],3*vertices[2]+1,3*vertices[2]+2};
    vector<int> cvDeletions(vertexMax);
    for (int ii = 0; ii < vertexMax; ++ii)
        cvDeletions[ii] = n_idx(ii,cellIndex);
    removeGPUArrayElement(vertexPositions,vpDeletions);
    removeGPUArrayElement(vertexMasses,vpDeletions);
    removeGPUArrayElement(vertexVelocities,vpDeletions);
    removeGPUArrayElement(vertexNeighbors,vnDeletions);
    removeGPUArrayElement(vertexCellNeighbors,vnDeletions);
    removeGPUArrayElement(cellVertexNum,cellIndex);
    removeGPUArrayElement(cellVertices,cvDeletions);

    removeGPUArrayElement(vertexEdgeFlips,vnDeletions);
    removeGPUArrayElement(vertexEdgeFlipsCurrent,vnDeletions);
    removeGPUArrayElement(cellSets,vnDeletions);
    removeGPUArrayElement(cellEdgeFlips,cellIndex);

    Nvertices -= 2;
    //phenomenal... let's handle the tag-to-index structures
    ittVertex.resize(Nvertices);
    ttiVertex.resize(Nvertices);
    vector<int> newTagToIdxV(Nvertices);
    vector<int> newIdxToTagV(Nvertices);
    int loopIndex = 0;
    int v1 = std::min(vertices[1],vertices[2]);
    int v2 = std::max(vertices[1],vertices[2]);
    for (int ii = 0; ii < Nvertices+2;++ii)
        {
        int vIdx = tagToIdxVertex[ii]; //vIdx is the current position of the vertex that was originally ii
        if (vIdx != v1 && vIdx != v2)
            {
            if (vIdx >= v1) vIdx = vIdx - 1;
            if (vIdx >= v2) vIdx = vIdx - 1;
            newTagToIdxV[loopIndex] = vIdx;
            loopIndex +=1;
            };
        };
    for (int ii = 0; ii < Nvertices; ++ii)
        newIdxToTagV[newTagToIdxV[ii]] = ii;
    tagToIdxVertex = newTagToIdxV;
    idxToTagVertex = newIdxToTagV;

    //finally, resize remaining stuff and call parent functions
    vertexForces.resize(Nvertices);
    displacements.resize(Nvertices);
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);

    initializeEdgeFlipLists(); //function call takes care of EdgeFlips and EdgeFlipsCurrent
    Simple2DActiveCell::cellDeath(cellIndex); //This call decrements Ncells by one
    n_idx = Index2D(vertexMax,Ncells);

    //computeGeometry();
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
Note that dParams does nothing
\post This function is meant to be called before the start of a new timestep. It should be immediately followed by a computeGeometry call
*/
void vertexModelBase::cellDivision(const vector<int> &parameters, const vector<Dscalar> &dParams)
    {
    //This function will first do some analysis to identify the cells and vertices involved
    //it will then call base class' cellDivision routine, and then update all needed data structures
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
    Box->minDist(vP.data[v1NextIdx],vP.data[v1idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV1Pos = vP.data[v1idx] + disp;
    Box->putInBoxReal(newV1Pos);
    Box->minDist(vP.data[v2NextIdx],vP.data[v2idx],disp);
    disp.x = 0.5*disp.x;
    disp.y = 0.5*disp.y;
    newV2Pos = vP.data[v2idx] + disp;
    Box->putInBoxReal(newV2Pos);

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

    //The Simple2DActiveCell routine will update Motility and cellDirectors,
    // it in turn calls the Simple2DCell routine, which grows its data structures and increment Ncells by one
    Simple2DActiveCell::cellDivision(parameters);

    Nvertices += 2;

    //additions to the spatial sorting vectors...
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

    //use the copy and grow mechanism where we need to actually set values
    growGPUArray(vertexPositions,2); //(nv)
    growGPUArray(vertexMasses,2); //(nv)
    growGPUArray(vertexVelocities,2); //(nv)
    growGPUArray(vertexNeighbors,6); //(3*nv)
    growGPUArray(vertexCellNeighbors,6); //(3*nv)
    growGPUArray(cellVertexNum,1); //(nc)
    growGPUArray(cellSets,6);//(3*nv)
    growGPUArray(cellEdgeFlips,1);
    //the index cellVertices array needs more care...
    vector<int>  cellVerticesVec;
    copyGPUArrayData(cellVertices,cellVerticesVec);
    cellVertices.resize(vertexMax*Ncells);
    //first, let's take care of the vertex positions, masses, and velocities
        {//arrayhandle scope
        ArrayHandle<Dscalar2> h_vp(vertexPositions);
        h_vp.data[Nvertices-2] = newV1Pos;
        h_vp.data[Nvertices-1] = newV2Pos;
        ArrayHandle<Dscalar2> h_vv(vertexVelocities);
        h_vv.data[Nvertices-2] = make_Dscalar2(0.0,0.0);
        h_vv.data[Nvertices-1] = make_Dscalar2(0.0,0.0);
        ArrayHandle<Dscalar> h_vm(vertexMasses);
        h_vm.data[Nvertices-2] = h_vm.data[v1idx];
        h_vm.data[Nvertices-1] = h_vm.data[v2idx];
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
