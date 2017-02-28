#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "selfPropelledParticleDynamics.cuh"

/** \file selfPropelledParticleDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup selfPropelledParticleDynamicsKernels
    @{
*/

/*!
  Each thread -- most likely corresponding to each cell -- is initialized with a different sequence
  of the same seed of a cudaRNG
*/
__global__ void initialize_sppRNG_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };


//!Call the kernel to initialize a different RNG for each particle
bool gpu_initialize_sppRNG(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    initialize_sppRNG_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration

