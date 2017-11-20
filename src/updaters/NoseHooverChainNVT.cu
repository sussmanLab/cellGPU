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
/** @} */ //end of group declaration

