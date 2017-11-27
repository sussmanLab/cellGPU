#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "setTotalLinearMomentum.cuh"

/*! \file setTotalLinearMomentum.cu
 Defines kernel callers and kernels for GPU calculations for shifting the total linear momentum of a system
*/

/*!
    \addtogroup updaterKernels
    @{
*/
/*!
Each thread updates the velocity of one particle
*/
__global__ void shift_momentum_kernel(
                                Dscalar2 *velocities,
                                Dscalar  *masses,
                                Dscalar2 pShift,
                                int      N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    velocities[idx] = velocities[idx]+(1.0/(N*masses[idx]))*pShift;
    return;
    };

//!simple shift of velocities
bool gpu_shift_momentum(
                    Dscalar2 *velocities,
                    Dscalar  *masses,
                    Dscalar2 pShift,
                    int       N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    shift_momentum_kernel<<<nblocks,block_size>>>(
                                velocities,
                                masses,
                                pShift,
                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration

