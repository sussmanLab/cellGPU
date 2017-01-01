#define ENABLE_CUDA

#include "avm2d.h"
#include "avm2d.cuh"

AVM2D::AVM2D(int n,Dscalar A0, Dscalar P0,bool reprod,bool initGPURNG)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... ",n);
    Reproducible = reprod;
    Initialize(n,initGPURNG);
    setCellPreferencesUniform(A0,P0);
    };

void AVM2D::setCellsVoronoiTesselation(int n)
    {
    //set number of cells, and a square box
    Ncells=n;
    CellPositions.resize(Ncells);
    Dscalar boxsize = sqrt((Dscalar)Ncells);
    Box.setSquare(boxsize,boxsize);

    //put cells in box randomly
    ArrayHandle<Dscalar2> h_p(CellPositions,access_location::host,access_mode::overwrite);
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
    VertexPositions.resize(Nvertices);
    ArrayHandle<Dscalar2> h_v(VertexPositions,access_location::host,access_mode::overwrite);

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

    //great... now, what is the maximum number of vertices for a cell?
    vertexMax = 0;
    for(PDT::Vertex_iterator vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
        {
        Vertex_circulator vc(vit);
        int base = vc ->info();
        int neighs = 1;
        ++vc;
        while(vc->info() != base)
            {
            neighs += 1;
            ++vc;
            };
        if (neighs > vertexMax) vertexMax = neighs;
        };
    vertexMax += 2;
    cout << "vM = " <<vertexMax << endl;
    //....now figure out indexing scheme to get voronoi vertices in some sensible way while simultaneously building the cell-vertex lists and vertex-vertes lists


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


