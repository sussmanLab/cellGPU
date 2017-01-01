#define ENABLE_CUDA

#include "avm2d.h"
#include "avm2d.cuh"

AVM2D::AVM2D(int n,Dscalar A0, Dscalar P0,bool reprod,bool initGPURNG)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... \n",n);
    Reproducible = reprod;
    Initialize(n,initGPURNG);
    setCellPreferencesUniform(A0,P0);
    };

void AVM2D::setCellsVoronoiTesselation(int n)
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

    map<PDT::Face_handle,int> faceToVoroIdx;
    int idx = 0;
    //first, ask CGAL for the circumcenter of the face, and add it to the list of vertices, and make a map between the iterator and the vertex idx
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
    ArrayHandle<int> h_vcn(vertexNeighbors,access_location::host,access_mode::overwrite);
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


    //randomly set vertex directors
    vertexDirectors.resize(Nvertices);
    ArrayHandle<Dscalar> h_vd(vertexDirectors,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        h_vd.data[ii] = 2.0*PI/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);

   };

//take care of all class initialization functions
void AVM2D::Initialize(int n,bool initGPU)
    {
    setCellsVoronoiTesselation(n);

    Timestep = 0;
    setDeltaT(0.01);

    AreaPeri.resize(Ncells);

    devStates.resize(Nvertices);
    if(initGPU)
        initializeCurandStates(1337,Timestep);
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
\param i the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void AVM2D::initializeCurandStates(int gs, int i)
    {
    ArrayHandle<curandState> d_cs(devStates,access_location::device,access_mode::overwrite);
    int globalseed = gs;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_curand(d_cs.data,Nvertices,i,globalseed);
    };

/*!
Very similar to the function in spv2d.cpp, but optimized since we already have some data structures (the voronoi vertices)
*/
void AVM2D::computeGeometryCPU()
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);

    //compute the geometry for each cell
    for (int i = 0; i < Ncells; ++i)
        {
        int neighs = h_nn.data[i];
        Dscalar2 cellPos = h_p.data[i];
        Dscalar2 vlast, vnext;
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        //compute the vertex position relative to the cell position
        Box.minDist(h_v.data[h_n.data[n_idx(neighs-1,i)]],cellPos,vlast);
        for (int nn = 0; nn < neighs; ++nn)
            {
            Box.minDist(h_v.data[h_n.data[n_idx(nn,i)]],cellPos,vnext);
            Varea += TriangleArea(vlast,vnext);
            Dscalar dx = vlast.x-vnext.x;
            Dscalar dy = vlast.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            vlast = vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

/*!
Very similar to the function in spv2d.cpp, but optimized since we already have some data structures (the voronoi vertices)
*/
void AVM2D::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
    ArrayHandle<int> d_nn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(cellVertices,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::readwrite);

    gpu_avm_geometry(
                    d_p.data,
                    d_v.data,
                    d_nn.data,
                    d_n.data,
                    d_AP.data,
                    Ncells,n_idx,Box);
    };

