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
#include "DelaunayCheckGPU.cuh"



__global__ void gpu_test_circumcircles_kernel(bool *d_redo,
                                              int *d_circumcircles,
                                              float2 *d_pt,
                                              unsigned int *d_cell_sizes,
                                              int *d_cell_idx,
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
//printf("idx = %i\t",idx);

    //the indices of particles forming the circumcircle
    int i1,i2,i3;
    i1 = d_circumcircles[3*idx];
    i2 = d_circumcircles[3*idx+1];
    i3 = d_circumcircles[3*idx+2];
//if (idx  < 1) printf("%i %i %i check",i1,i2,i3);
    //the vertex we will take to be the origin, and its cell position
    float2 v = d_pt[i1];
    float vz = 0.0;
    int ib,jb;
    ib=floor(v.x/boxsize);
    jb=floor(v.y/boxsize);

    
    
    float2 p1real = d_pt[i2];
    float2 p2real = d_pt[i3];

    float2 pt1,pt2;
    Box.minDist(p1real,v,pt1);
    Box.minDist(p2real,v,pt2);

    //get the circumcircle
    float2 Q;
    float rad;
    Circumcircle(vz,vz,pt1.x,pt1.y,pt2.x,pt2.y,
                    Q.x,Q.y,rad);

    //look through cells for other particles
    bool badParticle = false;
    float2 ptnew,toCenter; 
    int wcheck = ceil(rad/boxsize);
    if(wcheck > xsize/2) wcheck = xsize/2;
if(idx <1)
{
//    printf("(%f,%f), (%f,%f)\n",pt1.x,pt1.y,pt2.x,pt2.y);
//    printf("i1 %i, i2 %i, i3 %i, rad %f, cellsize %f wc = %i\n",i1,i2,i3,rad,boxsize,wcheck);

};
    rad = rad*rad;
    for (int ii = -wcheck; ii <= wcheck; ++ii)
        for (int jj = -wcheck; jj <= wcheck; ++jj)
            {
//if(idx <10) printf("%i\t",jj);
            if(badParticle) continue;

            int cx = (ib+ii);
            if(cx < 0) cx += xsize;
            if(cx >= xsize) cx -= xsize;
            int cy = (jb+jj);
            if(cy < 0) cx += ysize;
            if(cy >= xsize) cx -= ysize;

            int bin = ci(cx,cy);
            for (int pp = 0; pp < d_cell_sizes[bin]; ++pp)
                {
                int newidx = d_cell_idx[cli(pp,bin)];

                float2 pnreal = d_pt[newidx];
                Box.minDist(pnreal,v,ptnew);
                Box.minDist(ptnew,Q,toCenter);
//if(idx <10) printf("%i\t",newidx);
                //if it's in the circumcircle, check that its not one of the three points
                if(toCenter.x*toCenter.x+toCenter.y*toCenter.y < rad)
                    {
                    badParticle = true;
                    if (newidx == i1 || newidx == i2 || newidx ==i3) badParticle = false;
                    };

                };

            };// end loop over cells
    if (badParticle)
        {
printf("badparticle for idxs %i %i %i on threadidx%i\n",i1,i2,i3,idx);
        d_redo[idx] = true;
        d_redo[i1] = true;
        d_redo[i2] = true;
        d_redo[i3] = true;
        };

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
    
    cudaDeviceSynchronize();
    cout.flush();
    return cudaSuccess;
    };






#endif
