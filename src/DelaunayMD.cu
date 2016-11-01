#ifndef __DTEST_CU__
#define __DTEST_CU__

#define NVCC
#define ENABLE_CUDA
#define EPSILON 1e-12

#include <cuda_runtime.h>
#include "gpucell.cuh"
#include "indexer.h"
#include "gpubox.h"
#include "cu_functions.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayMD.cuh"



__global__ void gpu_move_particles_kernel(float2 *d_points,
                                          float2 *d_disp,
                                          int N,
                                          gpubox Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_points[idx].x = d_points[idx].x+d_disp[idx].x;
    d_points[idx].y = d_points[idx].y+d_disp[idx].y;
    Box.putInBox(d_points[idx]);
    return;
    };

bool gpu_move_particles(float2 *d_points,
                        float2 *d_disp,
                        int N,
                        gpubox &Box
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_move_particles_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_disp,
                                                N,
                                                Box
                                                );
    return cudaSuccess;
    };






#endif
