#define NVCC
#define ENABLE_CUDA

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

/*!
into the output vector put 0.5*m[i]*v[i]^2
*/
__global__ void NoseHooverChainNVT_prepare_KE_kernel(
                                Dscalar2 *velocities,
                                Dscalar  *masses,
                                Dscalar  *keArray,
                                int      N)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    keArray[idx] = 0.5*masses[idx]*(velocities[idx].x*velocities[idx].x+velocities[idx].y*velocities[idx].y);
    };

/*!
\param velocities Dscalar2 array of current velocities
\param masses Dscalar array of current masses
\param keArray Dscalar output array
\param N      the length of the arrays
\post keArray[idx] = 0.5*masses[idx]*(velocities[idx])^2
*/
bool gpu_prepare_KE_vector(Dscalar2   *velocities,
                              Dscalar *masses,
                              Dscalar *keArray,
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
                                Dscalar2 *velocities,
                                Dscalar  *kineticEnergyScaleFactor,
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
                    Dscalar2 *velocities,
                    Dscalar  *kineticEnergyScaleFactor,
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
                                Dscalar2 *velocities,
                                Dscalar2 *forces,
                                Dscalar  *masses,
                                Dscalar  deltaT,
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
                    Dscalar2 *velocities,
                    Dscalar2 *forces,
                    Dscalar  *masses,
                    Dscalar  deltaT,
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
