#define ENABLE_CUDA

#include "avm2d.h"
#include "avm2d.cuh"
#include "spv2d.h"

/*!
The constructor calls the Initialize function to take care of business, and
setCellPreferencesUniform to give all cells the same A_0 and p_0 values
*/
AVM2D::AVM2D(int n,Dscalar A0, Dscalar P0,bool reprod,bool initGPURNG,bool runSPVToInitialize)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... \n",n);
    Reproducible = reprod;
    GPUcompute=true;
    Initialize(n,initGPURNG,runSPVToInitialize);
    setCellPreferencesUniform(A0,P0);
    KA = 1.0;
    KP = 1.0;
    };

/*!
A function of convenience.... initialize cell positions and vertices by starting with the Delaunay
triangulations of a random point set. If you want something more regular, run the SPV mode for a few
timesteps to smooth out the random point set first.
\post After this is called, all topology data structures are initialized
*/
void AVM2D::setCellsVoronoiTesselation(int n, bool spvInitialize)
    {
    //set number of cells, and a square box
    Ncells=n;
    cellPositions.resize(Ncells);
    Dscalar boxsize = sqrt((Dscalar)Ncells);
    Box.setSquare(boxsize,boxsize);

    //put cells in box randomly
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Dscalar x =EPSILON+boxsize/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
        Dscalar y =EPSILON+boxsize/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
        if(x >=boxsize) x = boxsize-EPSILON;
        if(y >=boxsize) y = boxsize-EPSILON;
        h_p.data[ii].x = x;
        h_p.data[ii].y = y;
        };

    //use the SPV class to relax the initial configuration just a bit?
    if(spvInitialize)
        {
        SPV2D spv(Ncells,1.0,3.8,false);
        spv.setCPU(false);
        spv.setv0Dr(0.1,1.0);
        spv.setDeltaT(0.1);
        for (int ii = 0; ii < 10;++ii)
            spv.performTimestep();
        ArrayHandle<Dscalar2> h_pp(spv.points,access_location::host,access_mode::read);
        for (int ii = 0; ii < Ncells; ++ii)
            h_p.data[ii] = h_pp.data[ii];
        };

    //call CGAL to get Delaunay triangulation
    vector<pair<Point,int> > Psnew(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Psnew[ii]=make_pair(Point(h_p.data[ii].x,h_p.data[ii].y),ii);
        };
    Iso_rectangle domain(0.0,0.0,boxsize,boxsize);
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
    vertexMax += 2;
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
    //initialize edge flips to zero
    vertexEdgeFlips.resize(3*Nvertices);
    vertexEdgeFlipsCurrent.resize(3*Nvertices);
    ArrayHandle<int> h_vflip(vertexEdgeFlips,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vflipc(vertexEdgeFlipsCurrent,access_location::host,access_mode::overwrite);
    for (int i = 0; i < 3*Nvertices; ++i)
        {
        h_vflip.data[i]=0;
        h_vflipc.data[i]=0;
        }

    //randomly set vertex directors
    cellDirectors.resize(Ncells);
    ArrayHandle<Dscalar> h_cd(cellDirectors,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        h_cd.data[ii] = 2.0*PI/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
   };

/*!/
Take care of all class initialization functions, this involves setting arrays to the right size, etc.
*/
void AVM2D::Initialize(int n,bool initGPU,bool spvInitialize)
    {
    setCellsVoronoiTesselation(n,spvInitialize);

    Timestep = 0;
    setDeltaT(0.01);
    setT1Threshold(0.01);

    AreaPeri.resize(Ncells);

    devStates.resize(Nvertices);
    vertexForces.resize(Nvertices);
    vertexForceSets.resize(3*Nvertices);
    voroCur.resize(3*Nvertices);
    voroLastNext.resize(3*Nvertices);
    if(initGPU)
        initializeCurandStates(1337,Timestep);

    growCellVertexListAssist.resize(1);
    ArrayHandle<int> h_grow(growCellVertexListAssist,access_location::host,access_mode::overwrite);
    h_grow.data[0]=0;
    finishedFlippingEdges.resize(1);
    ArrayHandle<int> h_ffe(finishedFlippingEdges,access_location::host,access_mode::overwrite);
    h_ffe.data[0]=0;

    };

//set all cell area and perimeter preferences to uniform values
void AVM2D::setCellPreferencesUniform(Dscalar A0, Dscalar P0)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = A0;
        h_p.data[ii].y = P0;
        };
    };

/*!
when a T1 transition increases the maximum number of vertices around any cell in the system,
call this function first to copy over the cellVertices structure into a larger array
 */
void AVM2D::growCellVerticesList(int newVertexMax)
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
//    cellVertices = newCellVertices;
    cellVertices.swap(newCellVertices);
    };

/*!
\param i the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void AVM2D::initializeCurandStates(int gs, int i)
    {
    ArrayHandle<curandState> d_curandRNGs(devStates,access_location::device,access_mode::overwrite);
    int globalseed = gs;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_curand(d_curandRNGs.data,Ncells,i,globalseed);
    };

/*!
increment the time step, call either the CPU or GPU branch, depending on the state of
the GPUcompute flag
*/
void AVM2D::performTimestep()
    {
    Timestep += 1;
    if(GPUcompute)
        performTimestepGPU();
    else
        performTimestepCPU();
    };

/*!
Go through the parts of a timestep on the CPU
*/
void AVM2D::performTimestepCPU()
    {
    //compute the current area and perimeter of every cell
    computeGeometryCPU();
    //use this information to compute the net force on the vertices
    computeForcesCPU();
    //move the cells accordingly, and update the director of each cell
    displaceAndRotateCPU();
    //see if vertex motion leads to T1 transitions
    testAndPerformT1TransitionsCPU();
    //as a utility, one could compute the current "position" of the cells, but this is unnecessary
    //getCellPositionsCPU();
    };

/*!
go through the parts of a timestep on the GPU
*/
void AVM2D::performTimestepGPU()
    {
    //compute the current area and perimeter of every cell
    computeGeometryGPU();
    //use this information to compute the net force on the vertices
    computeForcesGPU();
    //move the cells accordingly, and update the director of each cell
    displaceAndRotateGPU();
    //see if vertex motion leads to T1 transitions...ONLY allow one transition per vertex and per cell per timestep
    testAndPerformT1TransitionsCPU();
    //as a utility, one could compute the current "position" of the cells, but this is unnecessary
    //getCellPositionsGPU();
    };

/*!
Very similar to the function in spv2d.cpp, but optimized since we already have some data structures
(the vertices)...compute the area and perimeter of the cells
*/
void AVM2D::computeGeometryCPU()
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
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex
*/
void AVM2D::computeForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its three cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        int cellIdx = h_vcn.data[fsidx];
        Dscalar Adiff = KA*(h_AP.data[cellIdx].x - h_APpref.data[cellIdx].x);
        Dscalar Pdiff = KP*(h_AP.data[cellIdx].y - h_APpref.data[cellIdx].y);
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetAVM is defined in inc/cu_functions.h
        computeForceSetAVM(vcur,vlast,vnext,Adiff,Pdiff,dEdv);

        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };
    };

/*!
Move every vertex according to the net force on it and its motility...CPU routine
For debugging, the random number generator gives the same sequence of "random" numbers every time.
For more random behavior, uncomment the "random_device rd;" line, and replace
mt19937 gen(rand());
with
mt19937 gen(rd());
*/
void AVM2D::displaceAndRotateCPU()
    {
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_cd(cellDirectors,access_location::host, access_mode::readwrite);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host, access_mode::readwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);

    //random_device rd;
    mt19937 gen(rand());
    normal_distribution<> normal(0.0,1.0);

    Dscalar directorx,directory;
    Dscalar2 disp;
    for (int i = 0; i < Nvertices; ++i)
        {
        //for uniform v0, the vertex director is the straight average of the directors of the cell neighbors
        directorx  = cos(h_cd.data[ h_vcn.data[3*i] ]);
        directorx += cos(h_cd.data[ h_vcn.data[3*i+1] ]);
        directorx += cos(h_cd.data[ h_vcn.data[3*i+2] ]);
        directorx /= 3.0;
        directory  = sin(h_cd.data[ h_vcn.data[3*i] ]);
        directory += sin(h_cd.data[ h_vcn.data[3*i+1] ]);
        directory += sin(h_cd.data[ h_vcn.data[3*i+2] ]);
        directory /= 3.0;
        //move vertices
        h_v.data[i].x += deltaT*(v0*directorx+h_f.data[i].x);
        h_v.data[i].y += deltaT*(v0*directory+h_f.data[i].y);
        Box.putInBoxReal(h_v.data[i]);
        };

    //update cell directors
    for (int i = 0; i < Ncells; ++i)
        h_cd.data[i] += normal(gen)*sqrt(2.0*deltaT*Dr);
    };

/*!
A utility function for the CPU routine. Given two vertex indices representing an edge that will undergo
a T1 transition, return in the pass-by-reference variables a helpful representation of the cells in the T1
and the vertices to be re-wired...see the comments in "testAndPerformT1TransitionsCPU" for what that representation is
*/
void AVM2D::getCellVertexSetForT1(int vertex1, int vertex2, int4 &cellSet, int4 &vertexSet, bool &growList)
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

    //Does the cell-vertex-neighbor data structure need to be bigger?
    if(h_cvn.data[cellSet.x] == vertexMax || h_cvn.data[cellSet.z] == vertexMax)
        growList = true;
    };

/*!
Test whether a T1 needs to be performed on any edge by simply checking if the edge length is beneath a threshold.
This function also performs the transition and maintains the auxiliary data structures
 */
void AVM2D::testAndPerformT1TransitionsCPU()
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
     IF v1 is above v2, the following is the convention (otherwise flip CW and CCW)
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
                Box.minDist(v1,v2,edge);
                if(norm(edge) < T1Threshold)
                    {
                    bool growCellVertexList = false;
                    getCellVertexSetForT1(vertex1,vertex2,cellSet,vertexSet,growCellVertexList);
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
                    Box.putInBoxReal(v1);
                    Box.putInBoxReal(v2);
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

                    //cell l gains v1 in between v2 and c
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
One would prefer the cell position to be defined as the centroid, requiring an additional computation of the cell area.
This may be implemented some day, but for now we define the cell position as the straight average of the vertex positions.
*/
void AVM2D::getCellPositionsCPU()
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
Very similar to the function in spv2d.cpp, but optimized since we already have some data structures (the vertices)
*/
void AVM2D::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,      access_location::device,access_mode::read);
    ArrayHandle<int>      d_cvn(cellVertexNum,       access_location::device,access_mode::read);
    ArrayHandle<int>      d_cv(cellVertices,         access_location::device,access_mode::read);
    ArrayHandle<int>      d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,             access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,       access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,            access_location::device,access_mode::overwrite);

    gpu_avm_geometry(
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
call kernels to (1) do force sets calculation, then (2) add them up
*/
void AVM2D::computeForcesGPU()
    {
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_fs(vertexForceSets,access_location::device, access_mode::overwrite);
    ArrayHandle<Dscalar2> d_f(vertexForces,access_location::device, access_mode::overwrite);

    int nForceSets = voroCur.getNumElements();
    gpu_avm_force_sets(
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    d_APpref.data,
                    d_fs.data,
                    nForceSets,
                    KA,
                    KP
                    );

    gpu_avm_sum_force_sets(
                    d_fs.data,
                    d_f.data,
                    Nvertices);
    };

/*!
Move every vertex according to the net force on it and its motility...GPU routine
*/
void AVM2D::displaceAndRotateGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device, access_mode::readwrite);
    ArrayHandle<Dscalar2> d_f(vertexForces,access_location::device, access_mode::read);
    ArrayHandle<Dscalar> d_cd(cellDirectors,access_location::device, access_mode::readwrite);
    ArrayHandle<curandState> d_cs(devStates,access_location::device,access_mode::read);
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);

    gpu_avm_displace_and_rotate(d_v.data,
                                d_f.data,
                                d_cd.data,
                                d_vcn.data,
                                d_cs.data,
                                v0,Dr,deltaT,
                                Box, Nvertices,Ncells);
    };
/*!
perform whatever check is desired for T1 transtions (here just a "is the edge too short")
and detect whether the edge needs to grow. If so, grow it!
*/
void AVM2D::testEdgesForT1GPU()
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
        gpu_avm_test_edges_for_T1(d_v.data,
                              d_vn.data,
                              d_vflip.data,
                              d_vcn.data,
                              d_cvn.data,
                              d_cv.data,
                              Box,
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

void AVM2D::flipEdgesGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_vn(vertexNeighbors,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_vflip(vertexEdgeFlips,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_vflipcur(vertexEdgeFlipsCurrent,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::readwrite);

    ArrayHandle<int> d_ffe(finishedFlippingEdges,access_location::device,access_mode::readwrite);

    gpu_avm_flip_edges(d_vflip.data,
                       d_vflipcur.data,
                       d_v.data,
                       d_vn.data,
                       d_vcn.data,
                       d_cvn.data,
                       d_cv.data,
                       d_ffe.data,
                       T1Threshold,
                       Box,
                       n_idx,
                       Nvertices,
                       Ncells);
    };

/*!
Because the cellVertexList might need to grow, it's convenient to break this into two parts
*/
void AVM2D::testAndPerformT1TransitionsGPU()
    {
    testEdgesForT1GPU();
    flipEdgesGPU();
    };

void AVM2D::getCellPositionsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::read);

    gpu_avm_get_cell_positions(d_p.data,
                               d_v.data,
                               d_cvn.data,
                               d_cv.data,
                               Ncells,
                               n_idx,
                               Box);
    };

