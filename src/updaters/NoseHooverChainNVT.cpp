#define ENABLE_CUDA

#include "NoseHooverChainNVT.h"
#include "NoseHooverChainNVT.cuh"
#include "utilities.cuh"
/*! \file NoseHooverChainNVT.cpp */

/*!
Initialize everything, by default setting the target temperature to unity.
Note that in the current set up the thermostate masses are automatically set by the target temperature, assuming \tau = 1
*/
NoseHooverChainNVT::NoseHooverChainNVT(int N, int M)
    {
    Timestep = 0;
    deltaT=0.01;
    GPUcompute=true;
    Ndof = N;
    displacements.resize(Ndof);
    keArray.resize(Ndof);
    keIntermediateReduction.resize(Ndof);
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
    kineticEnergyScaleFactor.resize(2);
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
    h_bv.data[0].w = 2.0 * (Ndof-2)*Temperature;
    for (int ii = 1; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].w = Temperature;
        };
    ArrayHandle<Dscalar> kes(kineticEnergyScaleFactor,access_location::host,access_mode::overwrite);
    kes.data[0] = h_bv.data[0].w;
    kes.data[1] = 1.0;
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
print out the current state of the bath: (pos, vel, accel, mass) for each element of the chain
*/
void NoseHooverChainNVT::reportBathData()
    {
    ArrayHandle<Dscalar4> bath(BathVariables);
    printf("position\tvelocity\tacceleration\tmass\n");
    for (int i = 0; i < BathVariables.getNumElements(); ++i)
        printf("%f\t%f\t%f\t%f\n",bath.data[i].x,bath.data[i].y,bath.data[i].z,bath.data[i].w);
    };

/*!
The implementation here closely follows algorithms 30 - 32 in Frenkel & Smit, generalized to the
case where the chain length is not necessarily always 2
*/
void NoseHooverChainNVT::integrateEquationsOfMotionCPU()
    {
    //We (i.e. Martyna et al., and Frenkel & Smit) use the Trotter formula approach to get time-reversible dynamics.
    {
    propagateChain();
    ArrayHandle<Dscalar> h_kes(kineticEnergyScaleFactor,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(State->returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] = h_kes.data[1]*h_v.data[ii];
    }
    propagatePositionsVelocities();
    {
    propagateChain();
    ArrayHandle<Dscalar> h_kes(kineticEnergyScaleFactor,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(State->returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] = h_kes.data[1]*h_v.data[ii];
    }
    };

/*!
The simple part of the algorithm actually updates the positions and velocities of the partices.
This is the step in which a force calculation is required.
*/
void NoseHooverChainNVT::propagatePositionsVelocities()
    {
    ArrayHandle<Dscalar> h_kes(kineticEnergyScaleFactor);
    h_kes.data[0] = 0.0;
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
        h_kes.data[0] += 0.5*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
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
    ArrayHandle<Dscalar> h_kes(kineticEnergyScaleFactor);
    Dscalar dt8 = 0.125*deltaT;
    Dscalar dt4 = 0.25*deltaT;
    Dscalar dt2 = 0.5*deltaT;

    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    ArrayHandle<Dscalar4> Bath(BathVariables);
    for (int ii = Nchain-1; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        Bath.data[ii].z = (Bath.data[ii-1].w*Bath.data[ii-1].y*Bath.data[ii-1].y-Temperature)/Bath.data[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        Dscalar ef = exp(-dt8*Bath.data[ii+1].y);
        Bath.data[ii].y *= ef;
        Bath.data[ii].y += Bath.data[ii].z*dt4;
        Bath.data[ii].y *= ef;
        };
    Bath.data[0].z = (2.0*h_kes.data[0] - 2.0*(Ndof-2)*Temperature)/Bath.data[0].w;
    Dscalar ef = exp(-dt8*Bath.data[1].y);
    Bath.data[0].y *= ef;
    Bath.data[0].y += Bath.data[0].z*dt4;
    Bath.data[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        Bath.data[ii].x += dt2*Bath.data[ii].y;

    //get the factor that will particle velocities
    h_kes.data[1] = exp(-dt2*Bath.data[0].y);
    //and pre-emptively update the kinetic energy
    h_kes.data[0] = h_kes.data[1]*h_kes.data[1]*h_kes.data[0];

    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    Bath.data[0].z = (2.0*h_kes.data[0] - 2.0*(Ndof-2)*Temperature)/Bath.data[0].w;
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
    //The kernel calling scheme. To avoid ridiculous numbers of brackets for array handle scoping,
    //we'll define helper functions

    //for now, let's update the chain variables on the CPU... profile later
    propagateChain(); // use data structure that holds [KE,s], update both.
    rescaleVelocitiesGPU(); //use the velocity vector and the [KE,s] data structure. Note that KE is already scaled by s^2 in the above step
    propagatePositionsVelocitiesGPU();
    calculateKineticEnergyGPU(); //get the kinetic energy into the [KE,s] data structure
    propagateChain();
    rescaleVelocitiesGPU();
    };

/*!
Do a multi-step dance to get the positions and velocities updated on the gpu branch
*/
void NoseHooverChainNVT::propagatePositionsVelocitiesGPU()
    {
    Dscalar deltaT2 = 0.5*deltaT;
    //first, we move particles according to their velocities
    State->moveDegreesOfFreedom(State->returnVelocities(),deltaT2);
    State->enforceTopology();
    State->computeForces();

    //Now we execute the second half of the time step.. first we need to update the velocities according to the forces and the masses
    {//array handle scope for the second half of the time step
    ArrayHandle<Dscalar2> d_f(State->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar> d_m(State->returnMasses(),access_location::device,access_mode::read);
    gpu_NoseHooverChainNVT_update_velocities(d_v.data,d_f.data,d_m.data,deltaT,Ndof);
    };
    State->moveDegreesOfFreedom(State->returnVelocities(),deltaT2);
    };

/*!
This combines multiple kernel calls. First we make a vector of kinetic energies per particle, then
we perform a parallel block reduction, and then a serial reduction
*/
void NoseHooverChainNVT::calculateKineticEnergyGPU()
    {
    {//array handle scope for keArray preparation
    ArrayHandle<Dscalar2> d_v(State->returnVelocities(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_m(State->returnMasses(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_keArray(keArray,access_location::device,access_mode::overwrite);
    gpu_prepare_KE_vector(d_v.data,d_m.data,d_keArray.data,Ndof);
    }

    {//array handle scope for parallel reduction
    ArrayHandle<Dscalar> d_keArray(keArray,access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar> d_keIntermediate(keIntermediateReduction,access_location::device,access_mode::overwrite);

    gpu_parallel_reduction(d_keArray.data,d_keIntermediate.data,d_kes.data,0,Ndof);
    }
    };

/*!
Simply call the velocity rescaling function...
*/
void NoseHooverChainNVT::rescaleVelocitiesGPU()
    {
    ArrayHandle<Dscalar2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::read);
    gpu_NoseHooverChainNVT_scale_velocities(d_v.data,d_kes.data,Ndof);
    };
