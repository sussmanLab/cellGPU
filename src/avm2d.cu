#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "avm2d.cuh"


//!initialize each thread with a different sequence of the same seed of a cudaRNG
__global__ void initialize_curand_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };


//!Call the kernel to initialize a different RNG for each particle
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    initialize_curand_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };


