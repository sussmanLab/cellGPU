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


__global__ void gpu_compute_cell_list_kernel(float2 *d_pt,
                                              unsigned int *d_cell_sizes,
                                              int *d_idx,
                                              int Np,
                                              unsigned int Nmax,
                                              int xsize,
                                              int ysize,
                                              float boxsize,
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

    float2 pos = d_pt[idx];

    int ibin = max(0,min(xsize-1,(int)floor(pos.x/boxsize)));
    int jbin = max(0,min(xsize-1,(int)floor(pos.y/boxsize)));
    int bin = ci(ibin,jbin);
    //if (bin > xsize*ysize) printf("(%f,%f) -- (%i,%i) in bin %i out of %i... %f \n",pos.x,pos.y,ibin,jbin,bin,xsize*ysize,boxsize);

    unsigned int offset = atomicAdd(&(d_cell_sizes[bin]), 1);
    //unsigned int offset = d_cell_sizes[bin];
//printf("bin %i  offset %i\n",bin,offset);
    //if (offset <= Nmax+1)
    if (offset <= d_assist[0]+1)
        {
        unsigned int write_pos = min(cli(offset, bin),cli.getNumElements());
        d_idx[write_pos] = idx;
//        atomicAdd(&(d_cell_sizes[bin]),1);
        //d_cell_sizes[bin] += 1;
        }
    else
        {
        //atomicMax(&(*Nmax), offset+1);
        //d_assist[0]=d_assist[0]+1;
        d_assist[0]=offset+1;
        //atomicAdd(&(d_assist[0]),1);
        d_assist[1]=1;
//        printf( "nmax violation ...nmax = %i   offset+1 = %i  \n",Nmax,offset+1);
        //atomicInc(&(*Nmax),1);
        //*Nmax = offset; //atomicMax(Nmax,offset+1);
        };

    return;
    };


bool gpu_compute_cell_list(float2 *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int &Nmax,
                                  int xsize,
                                  int ysize,
                                  float boxsize,
                                  voroguppy::gpubox &Box,
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
    //cout << "current NMax = " << nmax <<  endl;
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

    return cudaSuccess;
    }

#endif
