#define ENABLE_CUDA

#include "brownianParticleDynamics.h"
#include "brownianParticleDynamics.cuh"
/*! \file brownianParticleDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
brownianParticleDynamics::brownianParticleDynamics(int _N)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = true;
    mu = 1.0;
    Temperature = 1.0;
    Ndof = _N;
    RNGs.resize(Ndof);
    displacements.resize(Ndof);
    };

/*!
\param globalSeed the global seed to use
\param offset the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void brownianParticleDynamics::initializeGPURNGs(int globalSeed, int offset)
    {
    if(RNGs.getNumElements() != Ndof)
        RNGs.resize(Ndof);
    ArrayHandle<curandState> d_curandRNGs(RNGs,access_location::device,access_mode::overwrite);
    int globalseed = globalSeed;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_RNG(d_curandRNGs.data,Ndof,offset,globalseed);
    };

void brownianParticleDynamics::spatialSorting(const vector<int> &reIndexer)
    {
    reIndexing = cellModel->returnItt();
    reIndexRNG(RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable
*/
void brownianParticleDynamics::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    cellModel = dynamic_pointer_cast<Simple2DCell>(model);
    }

/*!
Advances brownian dynamics by one time step
*/
void brownianParticleDynamics::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    };

/*!
The straightforward GPU implementation
*/
void brownianParticleDynamics::integrateEquationsOfMotionGPU()
    {
    cellModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> d_f(cellModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);

    ArrayHandle<curandState> d_RNG(RNGs,access_location::device,access_mode::readwrite);

    gpu_brownian_eom_integration(d_f.data,
                 d_disp.data,
                 d_RNG.data,
                 Ndof,
                 deltaT,
                 mu,
                 Temperature);
    };//end array handle scope
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();
    };

/*!
The straightforward CPU implementation
*/
void brownianParticleDynamics::integrateEquationsOfMotionCPU()
    {
    cellModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> h_f(cellModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);

    normal_distribution<> normal(0.0,1.0);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        Dscalar randomNumber1,randomNumber2;
        if (Reproducible)
            {
            randomNumber1 = normal(gen);
            randomNumber2 = normal(gen);
            }
        else
            {
            randomNumber1 = normal(genrd);
            randomNumber2 = normal(genrd);
            }
        h_disp.data[ii].x = randomNumber1*sqrt(1.0*deltaT*Temperature*mu) + deltaT*mu*h_f.data[ii].x;
        h_disp.data[ii].y = randomNumber2*sqrt(1.0*deltaT*Temperature*mu) + deltaT*mu*h_f.data[ii].y;
        };
    };//end array handle scope
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();
    };
