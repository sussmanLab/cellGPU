#define ENABLE_CUDA

#include "selfPropelledParticleDynamics.h"
#include "selfPropelledParticleDynamics.cuh"
/*! \file selfPropelledParticleDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
selfPropelledParticleDynamics::selfPropelledParticleDynamics(int _N)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = true;
    mu = 1.0;
    Ndof = _N;
    };

/*!
\param globalSeed the global seed to use
\param offset the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void selfPropelledParticleDynamics::initializeRNGs(int globalSeed, int offset)
    {
    RNGs.resize(Ndof);
    ArrayHandle<curandState> d_curandRNGs(RNGs,access_location::device,access_mode::overwrite);
    int globalseed = globalSeed;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_sppRNG(d_curandRNGs.data,Ndof,offset,globalseed);
    };


/*!
Assign cell directors via a simple, reproducible RNG
*/
void selfPropelledParticleDynamics::setCellDirectorsRandomly()
    {
    cellDirectors.resize(Ndof);
    ArrayHandle<Dscalar> h_cd(cellDirectors,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < Ndof; ++ii)
        h_cd.data[ii] = 2.0*PI/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
    };

void selfPropelledParticleDynamics::spatialSorting(const vector<int> &reIndexer)
    {
    reIndexing = reIndexer;
    reIndexRNG(RNGs);
    reIndexArray(Motility);
    reIndexArray(cellDirectors);
    };

/*!
\param v0new the new value of velocity for all cells
\param drnew the new value of the rotational diffusion of cell directors for all cells
*/
void selfPropelledParticleDynamics::setv0Dr(Dscalar v0new,Dscalar drnew)
    {
    Motility.resize(Ndof);
    v0=v0new;
    Dr=drnew;
    if (true)
        {
        ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < Ndof; ++ii)
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
void selfPropelledParticleDynamics::setCellMotility(vector<Dscalar> &v0s,vector<Dscalar> &drs)
    {
    Motility.resize(dof);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < dof; ++ii)
        {
        h_mot.data[ii].x = v0s[ii];
        h_mot.data[ii].y = drs[ii];
        };
    };


/*!
Given a vector of forces, update the displacements array to the amounts needed to advance the simulation one step
*/
void selfPropelledParticleDynamics::integrateEquationsOfMotion(GPUArray<Dscalar2> &forces,
                                                               GPUArray<Dscalar2> &displacements)
    {
    if(GPUcompute)
        {//GPU routine
        }
    else
        {//CPU routine
        };

    };
