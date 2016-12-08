#ifndef __DMD_CU__
#define __DMD_CU__

#define NVCC
#define ENABLE_CUDA
#define EPSILON 1e-16

#include <cuda_runtime.h>
#include "gpucell.cuh"
#include "indexer.h"
#include "gpubox.h"
#include "cu_functions.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayMD.cuh"


__global__ void gpu_test_circumcenters_kernel(int *d_repair,
                                              int3 *d_circumcircles,
                                              Dscalar2 *d_pt,
                                              unsigned int *d_cell_sizes,
                                              int *d_cell_idx,
                                              int Nccs,
                                              int xsize,
                                              int ysize,
                                              Dscalar boxsize,
                                              gpubox Box,
                                              Index2D ci,
                                              Index2D cli,
                                              int *anyFail
                                              )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nccs)
        return;

    //the indices of particles forming the circumcircle
    int3 i1 = d_circumcircles[idx];
    //the vertex we will take to be the origin, and its cell position
    Dscalar2 v = d_pt[i1.x];
    int ib=floorf(v.x/boxsize);
    int jb=floorf(v.y/boxsize);

    Dscalar2 p1real = d_pt[i1.y];
    Dscalar2 p2real = d_pt[i1.z];

    Dscalar2 pt1,pt2;
    Box.minDist(p1real,v,pt1);
    Box.minDist(p2real,v,pt2);

    //get the circumcircle
    Dscalar2 Q;
    Dscalar rad;
    Circumcircle(pt1,pt2,Q,rad);

    //look through cells for other particles
    bool badParticle = false;
    Dscalar2 ptnew,toCenter;
    int wcheck = ceilf(rad/boxsize);

    if(wcheck > xsize/2) wcheck = xsize/2;
    rad *=1.0001;
    rad = rad*rad;
    for (int ii = -wcheck; ii <= wcheck; ++ii)
        {
        for (int jj = -wcheck; jj <= wcheck; ++jj)
            {
            int cx = (ib+ii);
            if(cx < 0) cx += xsize;
            if(cx >= xsize) cx -= xsize;
            int cy = (jb+jj);
            if(cy < 0) cy += ysize;
            if(cy >= ysize) cy -= ysize;

            int bin = ci(cx,cy);

            for (int pp = 0; pp < d_cell_sizes[bin]; ++pp)
                {
                int newidx = d_cell_idx[cli(pp,bin)];

                Dscalar2 pnreal = d_pt[newidx];
                Box.minDist(pnreal,v,ptnew);
                Box.minDist(ptnew,Q,toCenter);
                //if it's in the circumcircle, check that its not one of the three points
                if(toCenter.x*toCenter.x+toCenter.y*toCenter.y < rad)
                    {
                    if (newidx != i1.x && newidx != i1.y && newidx !=i1.z)
                        {
                        badParticle = true;
                        d_repair[newidx] = 1;
                        };
                    };

                };

            };
        };// end loop over cells

    if (badParticle)
        {
        *anyFail = 1;
        d_repair[i1.x] = 1;
        d_repair[i1.y] = 1;
        d_repair[i1.z] = 1;
        };

    return;
    };




__global__ void gpu_move_particles_kernel(Dscalar2 *d_points,
                                          Dscalar2 *d_disp,
                                          int N,
                                          gpubox Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_points[idx].x += d_disp[idx].x;
    d_points[idx].y += d_disp[idx].y;
    Box.putInBoxReal(d_points[idx]);
    return;
    };



bool gpu_test_circumcenters(int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            Dscalar2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            Dscalar boxsize,
                            gpubox &Box,
                            Index2D &ci,
                            Index2D &cli,
                            int &fail)
    {
    cudaError_t code;
    unsigned int block_size = 128;
    if (Nccs < 128) block_size = 32;
    unsigned int nblocks  = Nccs/block_size + 1;

    fail = 0;
    int *anyFail;
    cudaMalloc((void**)&anyFail,sizeof(int));
    cudaMemcpy(anyFail,&fail,sizeof(int),cudaMemcpyHostToDevice);


    gpu_test_circumcenters_kernel<<<nblocks,block_size>>>(
                            d_repair,
                       //     d_redo2,
                            d_ccs,
                            d_pt,
                            d_cell_sizes,
                            d_idx,
                            Nccs,
                            xsize,
                            ysize,
                            boxsize,
                            Box,
                            ci,
                            cli,
                            anyFail
                            );
    cudaMemcpy(&fail,anyFail,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(anyFail);


    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("testCircumcenters GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };




bool gpu_move_particles(Dscalar2 *d_points,
                        Dscalar2 *d_disp,
                        int N,
                        gpubox &Box
                        )
    {
    cudaError_t code;
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_move_particles_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_disp,
                                                N,
                                                Box
                                                );
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("moveParticle GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };

#endif
