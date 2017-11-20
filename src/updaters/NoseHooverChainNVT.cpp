#define ENABLE_CUDA

#include "NoseHooverChainNVT.h"
#include "NoseHooverChainNVT.cuh"
/*! \file NoseHooverChainNVT.cpp" */

/*!
Initialize everything, by default setting the target temperature to unity.
Note that in the current set up the thermostate masses are automatically set by the target temperature, assuming \tau = 1
*/
NoseHooverChainNVT::NoseHooverChainNVT(int N, int M)
    {
    Timestep = 0;
    deltaT=0.01;
    GPUcompute=true;
    kineticEnergy=0.0;
    Ndof = N;
    displacements.resize(Ndof);
    Nchain = M;
    BathVariables.resize(Nchain+1);
    ArrayHandle<Dscalar4> h_bv(BathVariables);
    //set the initial position and velocity of the thermostats to zero
    for (int ii = 0; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].x = 0.0;
        h_bv.data[ii].y = 0.0;
        h_bv.data[ii].z = 0.0;
        };
    setT(1.0);
    };

/*!
Set the target temperature to the specified value.
Additionally, use the observation in the Mol Phys paper to set the masses of the chain of thermostats
*/
void NoseHooverChainNVT::setT(Dscalar T)
    {
    Temperature = T;
    ArrayHandle<Dscalar4> h_bv(BathVariables);
    h_bv.data[0].w = 2 * Ndof*Temperature;
    kineticEnergy = 0.0;
    for (int ii = 1; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].w = Temperature;
        };
    };

/*!
Advance by one time step. Of note, for computational efficiency the topology is only updated on the
half-time steps (i.e., right before the instantaneous forces will to be computed). This means that
after each call to the simulation to "performTimestep()" there is no guarantee that the topology will
actually be up-to-date. Probably best to call enforceTopology just before saving or evaluating shape
outside of the normal timestep loops.
*/
void NoseHooverChainNVT::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (State->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = State->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        setT(Temperature); //the bath mass depends on the number of degrees of freedom
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
The implementation here closely follows algorithms 30 - 32 in Frenkel & Smit, generalized to the
case where the chain length is not necessarily always 2
*/
void NoseHooverChainNVT::integrateEquationsOfMotionCPU()
    {
    //We (i.e. Martyna et al., and Frenkel & Smit) use the Trotter formula approach to get time-reversible dynamics.
    propagateChain();
    propagatePositionsVelocities();
    propagateChain();
    };

/*!
The simple part of the algorithm actually updates the positions and velocities of the partices.
This is the step in which a force calculation is required.
*/
void NoseHooverChainNVT::propagatePositionsVelocities()
    {
    kineticEnergy = 0.0;
    Dscalar deltaT2 = 0.5*deltaT;
    {//scope for array handles in the first half of the time step
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_v(State->returnVelocities(),access_location::host,access_mode::read);
    for (int ii = 0; ii < Ndof; ++ii)
        h_disp.data[ii] = deltaT2*h_v.data[ii];
    };
    State->moveDegreesOfFreedom(displacements);
    State->enforceTopology();
    State->computeForces();

    {//array handle scope for the second half of the time step
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(State->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(State->returnVelocities(),access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar> h_m(State->returnMasses(),access_location::host,access_mode::read);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_v.data[ii] = h_v.data[ii] + (deltaT/h_m.data[ii])*h_f.data[ii];
        h_disp.data[ii] = deltaT2*h_v.data[ii];
        kineticEnergy += 0.5*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    };
    State->moveDegreesOfFreedom(displacements);

    };

/*!
The simple part of the algorithm partially updates the chain positions and velocities. It should be
called twice per time step
*/
void NoseHooverChainNVT::propagateChain()
    {
    Dscalar dt8 = 0.125*deltaT;
    Dscalar dt4 = 0.25*deltaT;
    Dscalar dt2 = 0.5*deltaT;

    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    ArrayHandle<Dscalar4> Bath(BathVariables);
    ArrayHandle<Dscalar2> h_v(State->returnVelocities());
    for (int ii = Nchain; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        Bath.data[ii].z = (Bath.data[ii-1].w*Bath.data[ii-1].y*Bath.data[ii-1].y-Temperature)/Bath.data[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        Dscalar ef = exp(-dt8*Bath.data[ii+1].y);
        Bath.data[ii].y *= ef;
        Bath.data[ii].y += Bath.data[ii].z*dt4;
        Bath.data[ii].y *= ef;
        };
    Bath.data[0].z = (kineticEnergy - 2.0*Ndof*Temperature)/Bath.data[0].w;
    Dscalar ef = exp(-dt8*Bath.data[1].y);
    Bath.data[0].y *= ef;
    Bath.data[0].y += Bath.data[0].z*dt4;
    Bath.data[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        Bath.data[ii].x += dt2*Bath.data[ii].y;

    //rescale particle velocities
    Dscalar s = exp(-dt2*Bath.data[0].y);
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] = s*h_v.data[ii];
    kineticEnergy = s*s*kineticEnergy;

    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    Bath.data[0].z = (kineticEnergy - 2.0*Ndof*Temperature)/Bath.data[0].w;
    ef = exp(-dt8*Bath.data[1].y);
    Bath.data[0].y *= ef;
    Bath.data[0].y += Bath.data[0].z*dt4;
    Bath.data[0].y *= ef;
    for (int ii = 1; ii < Nchain; ++ii)
        {
        Bath.data[ii].z = (Bath.data[ii-1].w*Bath.data[ii-1].y*Bath.data[ii-1].y-Temperature)/Bath.data[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        Dscalar ef = exp(-dt8*Bath.data[ii+1].y);
        Bath.data[ii].y *= ef;
        Bath.data[ii].y += Bath.data[ii].z*dt4;
        Bath.data[ii].y *= ef;
        };

    };

/*!
The GPU implementation of the identical algorithm done on the CPU
*/
void NoseHooverChainNVT::integrateEquationsOfMotionGPU()
    {
    };

