#ifndef __GPUCELL_CU__
#define __GPUCELL_CU__

#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "gpucell.cuh"
#include "indexer.h"
#include "gpubox.h"
#include <iostream>
#include <stdio.h>


__global__ void gpu_compute_cell_list_kernel(Dscalar2 *d_pt,
                                              unsigned int *d_cell_sizes,
                                              int *d_idx,
                                              int Np,
                                              unsigned int Nmax,
                                              int xsize,
                                              int ysize,
                                              Dscalar boxsize,
                                              gpubox Box,
                                              Index2D ci,
                                              Index2D cli,
                                              int *d_assist
                                              )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Np)
        return;

    Dscalar2 pos = d_pt[idx];

    int ibin = max(0,min(xsize-1,(int)floor(pos.x/boxsize)));
    int jbin = max(0,min(xsize-1,(int)floor(pos.y/boxsize)));
    int bin = ci(ibin,jbin);

    unsigned int offset = atomicAdd(&(d_cell_sizes[bin]), 1);
    if (offset <= d_assist[0]+1)
        {
        unsigned int write_pos = min(cli(offset, bin),cli.getNumElements()-1);
        d_idx[write_pos] = idx;
        }
    else
        {
        d_assist[0]=offset+1;
        d_assist[1]=1;
        };

    return;
    };

__global__ void gpu_zero_array_kernel(unsigned int *arr,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    arr[idx] = 0;
    return;
    };

__global__ void gpu_zero_array_kernel(int *arr,
                                      int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    arr[idx] = 0;
    return;
    };

/////
//Kernel callers
///

bool gpu_zero_array(unsigned int *arr,
                    int N
                    )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    return cudaSuccess;
    }

bool gpu_zero_array(int *arr,
                    int N
                    )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    return cudaSuccess;
    }




bool gpu_compute_cell_list(Dscalar2 *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int &Nmax,
                                  int xsize,
                                  int ysize,
                                  Dscalar boxsize,
                                  gpubox &Box,
                                  Index2D &ci,
                                  Index2D &cli,
                                  int *d_assist
                                  )
    {
    cudaError_t code;
    //optimize block size later
    unsigned int block_size = 128;
    if (Np < 128) block_size = 16;
    unsigned int nblocks  = Np/block_size + 1;


    unsigned int nmax = (unsigned int) Nmax;
    gpu_compute_cell_list_kernel<<<nblocks, block_size>>>(d_pt,
                                                          d_cell_sizes,
                                                          d_idx,
                                                          Np,
                                                          nmax,
                                                          xsize,
                                                          ysize,
                                                          boxsize,
                                                          Box,
                                                          ci,
                                                          cli,
                                                          d_assist
                                                          );
    //cudaThreadSynchronize();
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("compute_cell_list GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    }

#endif
