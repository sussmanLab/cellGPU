#define ENABLE_CUDA

#include "avm2d.h"

AVM2D::AVM2D(int n,Dscalar A0, Dscalar P0,bool reprod,bool initGPURNG)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... ",n);
    Reproducible = reprod;
    Initialize(n,initGPURNG);
    setCellPreferencesUniform(A0,P0);
    };

//take care of all class initialization functions
void AVM2D::Initialize(int n,bool initGPU)
    {
    Ncells=n;
    Nvertics = 6*Ncells;

    Timestep = 0;
    setDeltaT(0.01);

    AreaPeri.resize(Ncells);

    vertexDirectors.resize(Nvertices);
    ArrayHandle<Dscalar> h_vd(vertexDirectors,access_location::host, access_mode::overwrite);
    int randmax = 100000000;
    for (int ii = 0; ii < Nvertices; ++ii)
        h_vd.data[ii] = 2.0*PI/(Dscalar)(randmax)* (Dscalar)(rand()%randmax);
    
    devStates.resize(Nvertices);
    if(initGPU)
        initializeCurandStates(Timestep);
    };

//set all cell area and perimeter preferences to uniform values
void AVM2D::setCellPreferencesUniform(Dscalar A0, Dscalar P0)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
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
void SPV2D::initializeCurandStates(int i)
    {
    ArrayHandle<curandState> d_cs(devStates,access_location::device,access_mode::overwrite);

    int globalseed = 136;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_init_curand(d_cs.data,N,i,globalseed);

    };


