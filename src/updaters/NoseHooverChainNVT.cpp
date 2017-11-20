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
    Ndof = N;
    displacements.resize(Ndof);
    Nchain = M;
    BathVariables.resize(Nchain);
    ArrayHandle<Dscalar3> h_bv(BathVariables);
    //set the initial position and velocity of the thermostats to zero
    for (int ii = 0; ii < Nchain; ++ii)
        {
        h_bv.data[ii].x = 0.0;
        h_bv.data[ii].y = 0.0;
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
    ArrayHandle<Dscalar3> h_bv(BathVariables);
    h_bv.data[0].z = 2 * Ndof*Temperature;
    for (int ii = 1; ii < Nchain; ++ii) 
        h_bv.data[ii].z = Temperature;
    };

/*!
Advance by one time step
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
    };

/*!
The GPU implementation of the identical algorithm done on the CPU
*/
void NoseHooverChainNVT::integrateEquationsOfMotionGPU()
    {
    };

