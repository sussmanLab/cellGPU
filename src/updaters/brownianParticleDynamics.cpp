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
    noise.initialize(Ndof);
    displacements.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void brownianParticleDynamics::spatialSorting(const vector<int> &reIndexer)
    {
    //reIndexing = cellModel->returnItt();
    //reIndexRNG(noise.RNGs);
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
    if (cellModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = cellModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        };
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

    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

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

    for (int ii = 0; ii < Ndof; ++ii)
        {
        Dscalar randomNumber1 = noise.getRealNormal();
        Dscalar randomNumber2 = noise.getRealNormal();
        h_disp.data[ii].x = randomNumber1*sqrt(2.0*deltaT*Temperature*mu) + deltaT*mu*h_f.data[ii].x;
        h_disp.data[ii].y = randomNumber2*sqrt(2.0*deltaT*Temperature*mu) + deltaT*mu*h_f.data[ii].y;
        };
    };//end array handle scope
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();
    };
