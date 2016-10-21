#ifndef __DTEST_CU__
#define __DTEST_CU__

#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "gpucell.cuh"
#include "indexer.h"
#include "gpubox.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayCheckGPU.cuh"



__global__ void gpu_test_circumcircles_kernel(bool *d_redo,
                                              int *d_circumcircles,
                                              float2 *d_pt,
                                              unsigned int *d_cell_sizes,
                                              int *d_idx,
                                              int Np,
                                              int xsize,
                                              int ysize,
                                              float boxsize,
                                              gpubox Box,
                                              Index2D ci,
                                              Index2D cli
                                              )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Np)
        return;

    return;
    };

bool gpu_test_circumcircles(bool *d_redo,
                                  int *d_ccs,
                                  float2 *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int xsize,
                                  int ysize,
                                  float boxsize,
                                  gpubox &Box,
                                  Index2D &ci,
                                  Index2D &cli
                                  )
    {

    unsigned int block_size = 128;
    if (Np < 128) block_size = 16;
    unsigned int nblocks  = Np/block_size + 1;


    gpu_test_circumcircles_kernel<<<nblocks,block_size>>>(d_redo,
                                              d_ccs,
                                              d_pt,
                                              d_cell_sizes,
                                              d_idx,
                                              Np,
                                              xsize,
                                              ysize,
                                              boxsize,
                                              Box,
                                              ci,
                                              cli
                                              );

    return cudaSuccess;
    };






#endif
