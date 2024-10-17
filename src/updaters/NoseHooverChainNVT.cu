#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "NoseHooverChainNVT.cuh"

/*! \file NoseHooverChainNVT.cu
 Defines kernel callers and kernels for GPU calculations for integrating the NH equations of motion
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

__global__ void NoseHooverChainNVT_propagateChain_kernel(
                    double  *kineticEnergyScaleFactor,
                    double4 *bathVariables,
                    double Temperature,
                    double deltaT,
                    int Nchain,
                    int Ndof)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= 1)
        return;
    double dt8 = 0.125*deltaT;
    double dt4 = 0.25*deltaT;
    double dt2 = 0.5*deltaT;

    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    for (int ii = Nchain-1; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        bathVariables[ii].z = (bathVariables[ii-1].w*bathVariables[ii-1].y*bathVariables[ii-1].y-Temperature)/bathVariables[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        double ef = exp(-dt8*bathVariables[ii+1].y);
        bathVariables[ii].y *= ef;
        bathVariables[ii].y += bathVariables[ii].z*dt4;
        bathVariables[ii].y *= ef;
        };
    bathVariables[0].z = (2.0*kineticEnergyScaleFactor[0]/bathVariables[0].w - 1.0);
    double ef = exp(-dt8*bathVariables[1].y);
    bathVariables[0].y *= ef;
    bathVariables[0].y += bathVariables[0].z*dt4;
    bathVariables[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        bathVariables[ii].x += dt2*bathVariables[ii].y;

    //get the factor that will particle velocities
    kineticEnergyScaleFactor[1] = exp(-dt2*bathVariables[0].y);
    //and pre-emptively update the kinetic energy
    kineticEnergyScaleFactor[0] = kineticEnergyScaleFactor[1]*kineticEnergyScaleFactor[1]*kineticEnergyScaleFactor[0];

    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    bathVariables[0].z = (2.0*kineticEnergyScaleFactor[0]/bathVariables[0].w - 1.0);
    ef = exp(-dt8*bathVariables[1].y);
    bathVariables[0].y *= ef;
    bathVariables[0].y += bathVariables[0].z*dt4;
    bathVariables[0].y *= ef;
    for (int ii = 1; ii < Nchain; ++ii)
        {
        bathVariables[ii].z = (bathVariables[ii-1].w*bathVariables[ii-1].y*bathVariables[ii-1].y-Temperature)/bathVariables[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        double ef = exp(-dt8*bathVariables[ii+1].y);
        bathVariables[ii].y *= ef;
        bathVariables[ii].y += bathVariables[ii].z*dt4;
        bathVariables[ii].y *= ef;
        };
    };
                    

bool gpu_NoseHooverChainNVT_propagateChain(
                    double  *kineticEnergyScaleFactor,
                    double4 *bathVariables,
                    double Temperature,
                    double deltaT,
                    int Nchain,
                    int Ndof)
    {
    NoseHooverChainNVT_propagateChain_kernel<<<1,1>>>(
                                                kineticEnergyScaleFactor,
                                                bathVariables,
                                                Temperature,
                                                deltaT,
                                                Nchain,
                                                Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
into the output vector put 0.5*m[i]*v[i]^2
*/
__global__ void NoseHooverChainNVT_prepare_KE_kernel(
                                double2 *velocities,
                                double  *masses,
                                double  *keArray,
                                int      N)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    keArray[idx] = 0.5*masses[idx]*(velocities[idx].x*velocities[idx].x+velocities[idx].y*velocities[idx].y);
    };

/*!
\param velocities double2 array of current velocities
\param masses double array of current masses
\param keArray double output array
\param N      the length of the arrays
\post keArray[idx] = 0.5*masses[idx]*(velocities[idx])^2
*/
bool gpu_prepare_KE_vector(double2   *velocities,
                              double *masses,
                              double *keArray,
                              int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    NoseHooverChainNVT_prepare_KE_kernel<<<nblocks,block_size>>>(
                                                velocities,
                                                masses,
                                                keArray,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
Each thread scales the velocity of one particle by the second component of the helper array
*/
__global__ void NoseHooverChainNVT_scale_velocities_kernel(
                                double2 *velocities,
                                double  *kineticEnergyScaleFactor,
                                int      N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    velocities[idx].x *= kineticEnergyScaleFactor[1];
    velocities[idx].y *= kineticEnergyScaleFactor[1];
    return;
    };

//!Simply rescale every component of V by the scale factor
bool gpu_NoseHooverChainNVT_scale_velocities(
                    double2 *velocities,
                    double  *kineticEnergyScaleFactor,
                    int       N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    NoseHooverChainNVT_scale_velocities_kernel<<<nblocks,block_size>>>(
                                velocities,
                                kineticEnergyScaleFactor,
                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
Each thread updates the velocity of one particle
*/
__global__ void NoseHooverChainNVT_update_velocities_kernel(
                                double2 *velocities,
                                double2 *forces,
                                double  *masses,
                                double  deltaT,
                                int      N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    velocities[idx].x += (deltaT/masses[idx])*forces[idx].x;
    velocities[idx].y += (deltaT/masses[idx])*forces[idx].y;
    return;
    };

//!simple update of velocity based on force and mass
bool gpu_NoseHooverChainNVT_update_velocities(
                    double2 *velocities,
                    double2 *forces,
                    double  *masses,
                    double  deltaT,
                    int       N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    NoseHooverChainNVT_update_velocities_kernel<<<nblocks,block_size>>>(
                                velocities,
                                forces,
                                masses,
                                deltaT,
                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
