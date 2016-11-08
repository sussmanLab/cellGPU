using namespace std;

//definitions needed for DelaunayLoc, voroguppy namespace, Triangle, Triangle, and all GPU functions, respectively

#define EPSILON 1e-12
#define dbl float
#define REAL double
#define ANSI_DECLARATIONS
#define ENABLE_CUDA

#define PI 3.14159265358979323846

#include "spv2d.h"

SPV2D::SPV2D(int n)
    {
    printf("Initializing %i cells with random positions in a square box\n",n);
    Initialize(n);
    };

SPV2D::SPV2D(int n,float A0, float P0)
    {
    printf("Initializing %i cells with random positions in a square box\n",n);
    Initialize(n);
    setCellPreferencesUniform(A0,P0);
    };

void SPV2D::Initialize(int n)
    {
    setDeltaT(0.01);
    setDr(1.);
    initialize(n);
    forces.resize(n);
    AreaPeri.resize(n);
    cellDirectors.resize(n);
    ArrayHandle<float2> h_cd(cellDirectors,access_location::host, access_mode::overwrite);
    int randmax = 100000000;
    for (int ii = 0; ii < N; ++ii)
        {
        float theta = PI/(float)(randmax)* (float)(rand()%randmax);
        h_cd.data[ii].x = cos(theta);
        h_cd.data[ii].y = sin(theta);
        };
    };

void SPV2D::setCellPreferencesUniform(float A0, float P0)
    {
    AreaPeriPreferences.resize(N);
    ArrayHandle<float2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_p.data[ii].x = A0;
        h_p.data[ii].y = P0;
        };
    };

void SPV2D::computeSPVForces()
    {


    };

void SPV2D::performTimestep()
    {
    computeSPVForces();
    //vector of displacements is forces*timestep + v0's*timestep
    GPUArray<float2> ds; ds.resize(N);
    repel(ds,1e-3);

    movePoints(ds);
    //need to re-write a new movepoints that moves points and rotates cell directors
    testAndRepairTriangulation();

    };
