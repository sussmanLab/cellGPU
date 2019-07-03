#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "cellListGPU.cuh"
#include "indexer.h"
#include "gpubox.h"
#include <iostream>
#include <stdio.h>
/*! \file cellListGPU.cu */

/*!
    \addtogroup cellListGPUKernels
    @{
*/

/*!
  Assign particles to bins, keep track of the number of particles per bin, etc.
  */
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
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
