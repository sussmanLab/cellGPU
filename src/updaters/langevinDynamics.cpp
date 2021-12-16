#include "langevinDynamics.h"
#include "langevinDynamics.cuh"
#include "utilities.cuh"
/*! \file langevinDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
langevinDynamics::langevinDynamics(int _N, double _temperature, double _gamma, bool usegpu)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = usegpu;
    if(!GPUcompute)
        displacements.neverGPU=true;
    gamma = _gamma;
    Temperature = _temperature;
    Ndof = _N;
    noise.initializeGPURNG = GPUcompute;
    noise.initialize(Ndof);
    displacements.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void langevinDynamics::spatialSorting(const vector<int> &reIndexer)
    {
    //reIndexing = cellModel->returnItt();
    //reIndexRNG(noise.RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable
*/
void langevinDynamics::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    cellModel = dynamic_pointer_cast<Simple2DCell>(model);
    //set velocities uniformly randomly with the right temperature
        {
        Ndof = cellModel->getNumberOfDegreesOfFreedom();
        ArrayHandle<double2> h_v(cellModel->returnVelocities());
        double2 totalV; totalV.x=0.0;totalV.y=0.0;
        double sqrtT=sqrt(Temperature);
        for(int ii = 0; ii < Ndof; ++ii)
            {
            double randomNumber1 = noise.getRealUniform(-1.,1.)*sqrtT;
            double randomNumber2 = noise.getRealUniform(-1.,1.)*sqrtT;
            h_v.data[ii].x = randomNumber1;
            h_v.data[ii].y = randomNumber2;
            totalV.x+=randomNumber1; totalV.y += randomNumber2;
            }
        totalV.x = totalV.x / Ndof;
        totalV.y = totalV.y / Ndof;
        for(int ii = 0; ii < Ndof; ++ii)
            {
            h_v.data[ii].x -= totalV.x;
            h_v.data[ii].y -= totalV.y;
            }
        }
    }

/*!
Advances brownian dynamics by one time step
*/
void langevinDynamics::integrateEquationsOfMotion()
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
void langevinDynamics::integrateEquationsOfMotionGPU()
    {
    //B and O (update velocity, store half-step update for position, then update velocity!)
    cellModel->computeForces();
        {
        ArrayHandle<double2> d_f(cellModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
        ArrayHandle<double2> d_v(cellModel->returnVelocities(),access_location::device,access_mode::readwrite);
        ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

        gpu_langevin_BandO_operation(
                 d_v.data,
                 d_f.data,
                 d_disp.data,
                 d_RNG.data,
                 Ndof,
                 deltaT,
                 gamma,
                 Temperature);
        };//end array handle scope

    //A
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();

    //O -- already done!
    //A -- just re-set the displacement vector and move again
    gpu_copy_multipleOf_gpuarray(displacements, cellModel->returnVelocities(),0.5*deltaT);
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();

    //B
    cellModel->computeForces();
    gpu_add_multipleOf_gpuarray(cellModel->returnVelocities(),cellModel->returnForces(),0.5*deltaT,Ndof);
    };

/*!
The straightforward CPU implementation
*/
void langevinDynamics::integrateEquationsOfMotionCPU()
    {
    //B and O (update velocity, store half-step update for position, then update velocity!)
    cellModel->computeForces();
        {
        ArrayHandle<double2> h_f(cellModel->returnForces(),access_location::host,access_mode::read);
        ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);
        ArrayHandle<double2> h_v(cellModel->returnVelocities());
        for (int ii = 0; ii < Ndof; ++ii)
            {
            h_v.data[ii] = h_v.data[ii]+(0.5*deltaT)*h_f.data[ii];
            h_disp.data[ii] = (0.5*deltaT)*h_v.data[ii];
            double c1 = exp(-gamma*deltaT);
            double c2 = sqrt(Temperature)*sqrt(1.0-c1*c1);
            h_v.data[ii].x = c1*h_v.data[ii].x + noise.getRealNormal()*c2;
            h_v.data[ii].y = c1*h_v.data[ii].y + noise.getRealNormal()*c2;
            }
        }

    //A
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();

    //O -- already done!
    //A -- just re-set the displacement vector and move again
        {
        ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);
        ArrayHandle<double2> h_v(cellModel->returnVelocities());

        for (int ii = 0; ii < Ndof; ++ii)
            {
            h_disp.data[ii] = (0.5*deltaT)*h_v.data[ii];
            }
        }
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();

    //B
    cellModel->computeForces();
        {
        ArrayHandle<double2> h_f(cellModel->returnForces(),access_location::host,access_mode::read);
        ArrayHandle<double2> h_v(cellModel->returnVelocities());
        for (int ii = 0; ii < Ndof; ++ii)
            {
            h_v.data[ii] = h_v.data[ii] + (0.5*deltaT)*h_f.data[ii];
            }
        }
    };
