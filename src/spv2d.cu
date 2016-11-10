#ifndef __SPV2D_CU__
#define __SPV2D_CU__

#define NVCC
#define ENABLE_CUDA
#define EPSILON 1e-12

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "gpucell.cuh"
#include "spv2d.cuh"


#include "indexer.h"
#include "gpubox.h"
#include "cu_functions.h"
#include <iostream>
#include <stdio.h>

/*
__global__ void init_curand_kernel(unsigned long seed, curandState *state)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(seed,idx,0,&state[idx]);
    return;
    };
*/

__global__ void gpu_displace_and_rotate_kernel(float2 *d_points,
                                          float2 *d_force,
                                          float *d_directors,
                                          int N,
                                          float dt,
                                          float Dr,
                                          float v0,
                                          int seed,
//                                          curandState *states,
                                          gpubox Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    curandState_t randState;
    curand_init(seed,//seed first
                0,   // sequence -- only important for multiple cores
                0,   //offset. advance by sequence by 1 plus this value
                &randState);

    float dirx = cosf(d_directors[idx]);
    float diry = sinf(d_directors[idx]);
    //float angleDiff = curand_normal(&states[idx])*sqrt(2.0*dt*Dr);
    float angleDiff = curand_normal(&randState)*sqrt(2.0*dt*Dr);
    d_directors[idx] += angleDiff;

    float dx = dt*(v0*dirx + d_force[idx].x);
if (idx == 0) printf("x-displacement = %e\n",dx);
    d_points[idx].x += dt*(v0*dirx + d_force[idx].x);
    d_points[idx].y += dt*(v0*diry + d_force[idx].y);
    Box.putInBoxReal(d_points[idx]);
    return;
    };


//////////////
//kernel callers
//



/*
bool gpu_init_curand(curandState *states,
                    unsigned long seed,
                    int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    cudaMalloc((void **)&states,nblocks*block_size*sizeof(curandState) );
    init_curand_kernel<<<nblocks,block_size>>>(seed,states);
    return cudaSuccess;
    };
*/

bool gpu_displace_and_rotate(float2 *d_points,
                        float2 *d_force,
                        float  *d_directors,
                        int N,
                        float dt,
                        float Dr,
                        float v0,
                        int seed,
  //                      curandState *states,
                        gpubox &Box
                        )
    {
    cudaError_t code;
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_displace_and_rotate_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_force,
                                                d_directors,
                                                N,
                                                dt,
                                                Dr,
                                                v0,
                                                seed,
    //                                            states,
                                                Box
                                                );
    code = cudaGetLastError();
if(code!=cudaSuccess)
    printf("displaceAndRotate GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };


#endif
