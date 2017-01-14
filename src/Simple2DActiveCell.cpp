#define ENABLE_CUDA

#include "Simple2DActiveCell.h"
#include "Simple2DActiveCell.cuh"

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DActiveCell::Simple2DActiveCell() :
    Timestep(0), deltaT(0.01)
    {
    };

/*!
\param N the number of independent RNGs to initialize
\param i the value of the offset that should be sent to the cuda RNG...
\param gs the global seed to use
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void Simple2DActiveCell::initializeCurandStates(int N, int gs, int i)
    {
    cellRNGs.resize(N);
    ArrayHandle<curandState> d_curandRNGs(cellRNGs,access_location::device,access_mode::overwrite);
    int globalseed = gs;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_curand(d_curandRNGs.data,N,i,globalseed);
    };

/*!
Assign cell directors via a simple, reproducible RNG
*/
void Simple2DActiveCell::setCellDirectorsRandomly()
    {
    ArrayHandle<Dscalar> h_cd(cellDirectors,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Dscalar theta = 2.0*PI/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
        h_cd.data[ii] = theta;
        };
    };
    
/*!
\param v0new the new value of velocity for all cells
\param drnew the new value of the rotational diffusion of cell directors for all cells
*/
void Simple2DActiveCell::setv0Dr(Dscalar v0new,Dscalar drnew)
    {
    Motility.resize(Ncells);
    v0=v0new;
    Dr=drnew;
    if (true)
        {
        ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < Ncells; ++ii)
            {
            h_mot.data[ii].x = v0new;
            h_mot.data[ii].y = drnew;
            };
        };
    };

/*!
\param v0s the per-particle vector of what all velocities will be
\param drs the per-particle vector of what all rotational diffusions will be
*/
void Simple2DActiveCell::setCellMotility(vector<Dscalar> &v0s,vector<Dscalar> &drs)
    {
    Motility.resize(Ncells);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_mot.data[ii].x = v0s[ii];
        h_mot.data[ii].y = drs[ii];
        };
    };

