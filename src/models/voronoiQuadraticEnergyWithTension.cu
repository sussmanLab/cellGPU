#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "cellListGPU.cuh"
#include "voronoiQuadraticEnergyWithTension.cuh"

#include "indexer.h"
#include "gpubox.h"
#include "functions.h"
#include <iostream>
#include <stdio.h>
#include "Matrix.h"
/*! \file voronoiQuadraticEnergyWithTension.cu */

/*!
\file A file defining some global kernels for use in the spv2d Tension class
*/

/*!
    \addtogroup spvKernels
    @{
*/

//!the force on a particle is decomposable into the force contribution from each of its voronoi vertices...calculate those sets of forces with an additional tension term between cells of different type
__global__ void gpu_VoronoiTension_force_sets_kernel(const Dscalar2* __restrict__ d_points,
                                          const Dscalar2* __restrict__ d_AP,
                                          const Dscalar2* __restrict__ d_APpref,
                                          const int2* __restrict__ d_delSets,
                                          const int* __restrict__ d_delOther,
                                          const Dscalar2* __restrict__ d_vc,
                                          const Dscalar4* __restrict__ d_vln,
                                          Dscalar2* __restrict__ d_forceSets,
                                          const int2* __restrict__ d_nidx,
                                          const int* __restrict__ d_cellTypes,
                                          const Dscalar* __restrict__ d_tensionMatrix,
                                          Index2D cellTypeIndexer,
                                          Dscalar   KA,
                                          Dscalar   KP,
                                          int     computations,
                                          Index2D n_idx,
                                          gpubox Box
                                        )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= computations)
        return;

    //which particle are we evaluating, and which neighbor
    int pidx = d_nidx[tidx].x;
    int nn = d_nidx[tidx].y;
    int nidx=n_idx(nn,pidx);

    //Great...access the Delaunay neighbors and the relevant other point
    Dscalar2 pi   = d_points[pidx];

    int2 neighs = d_delSets[nidx];
    int neighOther = d_delOther[nidx];
    Dscalar2 rij, rik,pno;

    Box.minDist(d_points[neighs.x],pi,rij);
    Box.minDist(d_points[neighs.y],pi,rik);
    Box.minDist(d_points[neighOther],pi,pno);

    //first, compute the derivative of the main voro point w/r/t pidx's position
    Matrix2x2 dhdr;
    getdhdr(dhdr,rij,rik);

    //finally, compute all of the forces
    //pnm1 is rij, pn1 is rik
    Dscalar2 vlast,vcur,vnext,vother;
    vcur = d_vc[nidx];
    Dscalar4 vvv = d_vln[nidx];
    vlast.x = vvv.x; vlast.y = vvv.y;
    vnext.x = vvv.z; vnext.y = vvv.w;
    Circumcenter(rij,rik,pno,vother);


    Dscalar2 dAdv,dPdv,dTdv;
    Dscalar2 dEdv;
    Dscalar  Adiff, Pdiff;
    Dscalar2 dlast, dnext,dcl,dnc;
    Dscalar  dlnorm,dnnorm,dclnorm,dncnorm;
    bool Tik = false;
    bool Tij = false;
    bool Tjk = false;
    int typeI, typeJ, typeK;
    typeI = d_cellTypes[pidx];
    typeK = d_cellTypes[neighs.y];
    typeJ = d_cellTypes[neighs.x];
    if (typeI != typeK) Tik = true;
    if (typeI != typeJ) Tij = true;
    if (typeJ != typeK) Tjk = true;
    //neighs.y is "baseNeigh" of cpu routing... neighs.x is "otherNeigh"....neighOther is "DT_other_idx"

    //self terms
    dAdv.x = 0.5*(vlast.y-vnext.y);
    dAdv.y = 0.5*(vnext.x-vlast.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y=vlast.y-vcur.y;
    dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
    if(dlnorm < THRESHOLD)
        dlnorm = THRESHOLD;

    //save a few of these differences for later...
    dcl.x = -dlast.x;dcl.y = -dlast.y;
    dnc.x=-dnext.x;dnc.y=-dnext.y;
    dclnorm=dlnorm;
    dncnorm=dnnorm;

    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tik)
        {
        dTdv.x -= d_tensionMatrix[cellTypeIndexer(typeK,typeI)]*dnext.x/dnnorm;
        dTdv.y -= d_tensionMatrix[cellTypeIndexer(typeK,typeI)]*dnext.y/dnnorm;
        };
    if(Tij)
        {
        dTdv.x += d_tensionMatrix[cellTypeIndexer(typeJ,typeI)]*dlast.x/dlnorm;
        dTdv.y += d_tensionMatrix[cellTypeIndexer(typeJ,typeI)]*dlast.y/dlnorm;
        };

    Adiff = KA*(d_AP[pidx].x - d_APpref[pidx].x);
    Pdiff = KP*(d_AP[pidx].y - d_APpref[pidx].y);

    //defer a global factor of two to the very end...saves six multiplications...
    dEdv.x  =  Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*dTdv.x;
    dEdv.y  =  Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*dTdv.y;

    //other terms...k first...
    dAdv.x = 0.5*(vnext.y-vother.y);
    dAdv.y = 0.5*(vother.x-vnext.x);
    dnext.x = vcur.x-vother.x;
    dnext.y = vcur.y-vother.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
    dPdv.x = dnc.x/dncnorm - dnext.x/dnnorm;
    dPdv.y = dnc.y/dncnorm - dnext.y/dnnorm;
    Adiff = KA*(d_AP[neighs.y].x - d_APpref[neighs.y].x);
    Pdiff = KP*(d_AP[neighs.y].y - d_APpref[neighs.y].y);
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tik)
        {
        dTdv.x += d_tensionMatrix[cellTypeIndexer(typeK,typeI)]*dnc.x/dncnorm;
        dTdv.y += d_tensionMatrix[cellTypeIndexer(typeK,typeI)]*dnc.y/dncnorm;
        };
    if(Tjk)
        {
        dTdv.x -= d_tensionMatrix[cellTypeIndexer(typeK,typeJ)]*dnext.x/dnnorm;
        dTdv.y -= d_tensionMatrix[cellTypeIndexer(typeK,typeJ)]*dnext.y/dnnorm;
        };

    dEdv.x  += Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*dTdv.x;
    dEdv.y  += Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*dTdv.y;

    //...and then j
    dAdv.x = 0.5*(vother.y-vlast.y);
    dAdv.y = 0.5*(vlast.x-vother.x);
    dlast.x = -dnext.x;
    dlast.y = -dnext.y;
    dlnorm = dnnorm;
    dPdv.x = dlast.x/dlnorm - dcl.x/dclnorm;
    dPdv.y = dlast.y/dlnorm - dcl.y/dclnorm;
    Adiff = KA*(d_AP[neighs.x].x - d_APpref[neighs.x].x);
    Pdiff = KP*(d_AP[neighs.x].y - d_APpref[neighs.x].y);
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tij)
        {
        dTdv.x -= d_tensionMatrix[cellTypeIndexer(typeJ,typeI)]*dcl.x/dclnorm;
        dTdv.y -= d_tensionMatrix[cellTypeIndexer(typeJ,typeI)]*dcl.y/dclnorm;
        };
    if(Tjk)
        {
        dTdv.x += d_tensionMatrix[cellTypeIndexer(typeK,typeJ)]*dlast.x/dlnorm;
        dTdv.y += d_tensionMatrix[cellTypeIndexer(typeK,typeJ)]*dlast.y/dlnorm;
        };

    dEdv.x  +=  Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*dTdv.x;
    dEdv.y  +=  Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*dTdv.y;

    dEdv.x *= 2.0;
    dEdv.y *= 2.0;

    d_forceSets[nidx] = dEdv*dhdr;

    return;
    };

//!the force on a particle is decomposable into the force contribution from each of its voronoi vertices...calculate those sets of forces with an additional tension term between cells of different type
__global__ void gpu_VoronoiSimpleTension_force_sets_kernel(const Dscalar2* __restrict__ d_points,
                                          const Dscalar2* __restrict__ d_AP,
                                          const Dscalar2* __restrict__ d_APpref,
                                          const int2* __restrict__ d_delSets,
                                          const int* __restrict__ d_delOther,
                                          const Dscalar2* __restrict__ d_vc,
                                          const Dscalar4* __restrict__ d_vln,
                                          Dscalar2* __restrict__ d_forceSets,
                                          const int2* __restrict__ d_nidx,
                                          const int* __restrict__ d_cellTypes,
                                          Dscalar   KA,
                                          Dscalar   KP,
                                          Dscalar   gamma,
                                          int     computations,
                                          Index2D n_idx,
                                          gpubox Box
                                        )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= computations)
        return;

    //which particle are we evaluating, and which neighbor
    int pidx = d_nidx[tidx].x;
    int nn = d_nidx[tidx].y;
    int nidx=n_idx(nn,pidx);

    //Great...access the Delaunay neighbors and the relevant other point
    Dscalar2 pi   = d_points[pidx];

    int2 neighs = d_delSets[nidx];
    int neighOther = d_delOther[nidx];
    Dscalar2 rij, rik,pno;

    Box.minDist(d_points[neighs.x],pi,rij);
    Box.minDist(d_points[neighs.y],pi,rik);
    Box.minDist(d_points[neighOther],pi,pno);

    //first, compute the derivative of the main voro point w/r/t pidx's position
    Matrix2x2 dhdr;
    getdhdr(dhdr,rij,rik);

    //finally, compute all of the forces
    //pnm1 is rij, pn1 is rik
    Dscalar2 vlast,vcur,vnext,vother;
    vcur = d_vc[nidx];
    Dscalar4 vvv = d_vln[nidx];
    vlast.x = vvv.x; vlast.y = vvv.y;
    vnext.x = vvv.z; vnext.y = vvv.w;
    Circumcenter(rij,rik,pno,vother);


    Dscalar2 dAdv,dPdv,dTdv;
    Dscalar2 dEdv;
    Dscalar  Adiff, Pdiff;
    Dscalar2 dlast, dnext,dcl,dnc;
    Dscalar  dlnorm,dnnorm,dclnorm,dncnorm;
    bool Tik = false;
    bool Tij = false;
    bool Tjk = false;
    if (d_cellTypes[pidx] != d_cellTypes[neighs.y]) Tik = true;
    if (d_cellTypes[pidx] != d_cellTypes[neighs.x]) Tij = true;
    if (d_cellTypes[neighs.y] != d_cellTypes[neighs.x]) Tjk = true;
    //neighs.y is "baseNeigh" of cpu routing... neighs.x is "otherNeigh"....neighOther is "DT_other_idx"

    //self terms
    dAdv.x = 0.5*(vlast.y-vnext.y);
    dAdv.y = 0.5*(vnext.x-vlast.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y=vlast.y-vcur.y;
    dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
    if(dlnorm < THRESHOLD)
        dlnorm = THRESHOLD;

    //save a few of these differences for later...
    dcl.x = -dlast.x;dcl.y = -dlast.y;
    dnc.x=-dnext.x;dnc.y=-dnext.y;
    dclnorm=dlnorm;
    dncnorm=dnnorm;

    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tik)
        {
        dTdv.x -= dnext.x/dnnorm;
        dTdv.y -= dnext.y/dnnorm;
        };
    if(Tij)
        {
        dTdv.x += dlast.x/dlnorm;
        dTdv.y += dlast.y/dlnorm;
        };

    Adiff = KA*(d_AP[pidx].x - d_APpref[pidx].x);
    Pdiff = KP*(d_AP[pidx].y - d_APpref[pidx].y);

    //defer a global factor of two to the very end...saves six multiplications...
    dEdv.x  =  Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*gamma*dTdv.x;
    dEdv.y  =  Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*gamma*dTdv.y;

    //other terms...k first...
    dAdv.x = 0.5*(vnext.y-vother.y);
    dAdv.y = 0.5*(vother.x-vnext.x);
    dnext.x = vcur.x-vother.x;
    dnext.y = vcur.y-vother.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
    dPdv.x = dnc.x/dncnorm - dnext.x/dnnorm;
    dPdv.y = dnc.y/dncnorm - dnext.y/dnnorm;
    Adiff = KA*(d_AP[neighs.y].x - d_APpref[neighs.y].x);
    Pdiff = KP*(d_AP[neighs.y].y - d_APpref[neighs.y].y);
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tik)
        {
        dTdv.x += dnc.x/dncnorm;
        dTdv.y += dnc.y/dncnorm;
        };
    if(Tjk)
        {
        dTdv.x -= dnext.x/dnnorm;
        dTdv.y -= dnext.y/dnnorm;
        };

    dEdv.x  += Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*gamma*dTdv.x;
    dEdv.y  += Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*gamma*dTdv.y;

    //...and then j
    dAdv.x = 0.5*(vother.y-vlast.y);
    dAdv.y = 0.5*(vlast.x-vother.x);
    dlast.x = -dnext.x;
    dlast.y = -dnext.y;
    dlnorm = dnnorm;
    dPdv.x = dlast.x/dlnorm - dcl.x/dclnorm;
    dPdv.y = dlast.y/dlnorm - dcl.y/dclnorm;
    Adiff = KA*(d_AP[neighs.x].x - d_APpref[neighs.x].x);
    Pdiff = KP*(d_AP[neighs.x].y - d_APpref[neighs.x].y);
    dTdv.x = 0.0; dTdv.y = 0.0;
    if(Tij)
        {
        dTdv.x -= dcl.x/dclnorm;
        dTdv.y -= dcl.y/dclnorm;
        };
    if(Tjk)
        {
        dTdv.x += dlast.x/dlnorm;
        dTdv.y += dlast.y/dlnorm;
        };

    dEdv.x  +=  Adiff*dAdv.x + Pdiff*dPdv.x + 0.5*gamma*dTdv.x;
    dEdv.y  +=  Adiff*dAdv.y + Pdiff*dPdv.y + 0.5*gamma*dTdv.y;

    dEdv.x *= 2.0;
    dEdv.y *= 2.0;

    d_forceSets[nidx] = dEdv*dhdr;

    return;
    };


//!Call the kernel to compute force sets with a generic matrix of surface tensions between types
bool gpu_VoronoiTension_force_sets(Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_delOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2   *d_nidx,
                    int    *d_cellTypes,
                    Dscalar *d_tensionMatrix,
                    Index2D &cellTypeIndexer,
                    Dscalar  KA,
                    Dscalar  KP,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    )
    {
    unsigned int block_size = 128;
    if (NeighIdxNum < 128) block_size = 32;
    unsigned int nblocks  = NeighIdxNum/block_size + 1;

    gpu_VoronoiTension_force_sets_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_AP,
                                                d_APpref,
                                                d_delSets,
                                                d_delOther,
                                                d_vc,
                                                d_vln,
                                                d_forceSets,
                                                d_nidx,
                                                d_cellTypes,
                                                d_tensionMatrix,
                                                cellTypeIndexer,
                                                KA,
                                                KP,
                                                NeighIdxNum,
                                                n_idx,
                                                Box
                                                );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to compute force sets with additional (uniform) tension terms
bool gpu_VoronoiSimpleTension_force_sets(Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_delOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2   *d_nidx,
                    int    *d_cellTypes,
                    Dscalar  KA,
                    Dscalar  KP,
                    Dscalar  gamma,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    )
    {
    unsigned int block_size = 128;
    if (NeighIdxNum < 128) block_size = 32;
    unsigned int nblocks  = NeighIdxNum/block_size + 1;

    gpu_VoronoiSimpleTension_force_sets_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_AP,
                                                d_APpref,
                                                d_delSets,
                                                d_delOther,
                                                d_vc,
                                                d_vln,
                                                d_forceSets,
                                                d_nidx,
                                                d_cellTypes,
                                                KA,
                                                KP,
                                                gamma,
                                                NeighIdxNum,
                                                n_idx,
                                                Box
                                                );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;

    };
/** @} */ //end of group declaration
