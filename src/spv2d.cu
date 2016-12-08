#ifndef __SPV2D_CU__
#define __SPV2D_CU__

#define NVCC
#define ENABLE_CUDA
#define EPSILON 1e-16
#define THRESHOLD 1e-16

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "gpucell.cuh"
#include "spv2d.cuh"

#include "indexer.h"
#include "gpubox.h"
#include "cu_functions.h"
#include <iostream>
#include <stdio.h>
#include "Matrix.h"


__global__ void init_curand_kernel(curandState *state, int N,int Timestep)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(1337,idx,Timestep,&state[idx]);
    return;
    };


__global__ void gpu_sum_forces_kernel(const Dscalar2* __restrict__ d_forceSets,
                                      Dscalar2* __restrict__ d_forces,
                                      const int* __restrict__      d_nn,
                                      int     N,
                                      Index2D n_idx
                                     )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int neigh = d_nn[idx];
    Dscalar2 temp;
    temp.x=0.0;temp.y=0.0;
    for (int nn = 0; nn < neigh; ++nn)
        {
        Dscalar2 val = d_forceSets[n_idx(nn,idx)];
        temp.x+=val.x;
        temp.y+=val.y;
        };

    d_forces[idx]=temp;

    };

__global__ void gpu_sum_forces_with_exclusions_kernel(const Dscalar2* __restrict__ d_forceSets,
                                      Dscalar2* __restrict__ d_forces,
                                      Dscalar2* __restrict__ d_external_forces,
                                      const int* __restrict__ d_exes,
                                      const int* __restrict__ d_nn,
                                      int     N,
                                      Index2D n_idx
                                     )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int neigh = d_nn[idx];
    Dscalar2 temp;
    temp.x=0.0;temp.y=0.0;
    for (int nn = 0; nn < neigh; ++nn)
        {
        Dscalar2 val = d_forceSets[n_idx(nn,idx)];
        temp.x+=val.x;
        temp.y+=val.y;
        };
    if (d_exes[idx] ==0)
        {
        d_forces[idx]=temp;
        d_external_forces[idx] = make_Dscalar2(0.0,0.0);
        }
    else
        {
        d_forces[idx]=make_Dscalar2(0.0,0.0);
        d_external_forces[idx] = make_Dscalar2(-temp.x,-temp.y);
        };

    };

__global__ void gpu_force_sets_kernel(const Dscalar2* __restrict__ d_points,
                                      const Dscalar2* __restrict__ d_AP,
                                      const Dscalar2*  __restrict__ d_APpref,
                                      const int4* __restrict__ d_delSets,
                                      const int* __restrict__ d_delOther,
                                      const Dscalar2* __restrict__ d_vc,
                                      const Dscalar4* __restrict__ d_vln,
                                      Dscalar2* __restrict__ d_forceSets,
                                      const int2* __restrict__ d_nidx,
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

    //Great...access the four Delaunay neighbors and the relevant fifth point
    Dscalar2 pi, pnm2,rij, rik,pn2,pno;
    int4 neighs;
    pi   = d_points[pidx];

    neighs = d_delSets[n_idx(nn,pidx)];

    Box.minDist(d_points[neighs.x],pi,pnm2);
    Box.minDist(d_points[neighs.y],pi,rij);
    Box.minDist(d_points[neighs.z],pi,rik);
    Box.minDist(d_points[neighs.w],pi,pn2);
    Box.minDist(d_points[d_delOther[n_idx(nn,pidx)]],pi,pno);

    //first, compute the derivative of the main voro point w/r/t pidx's position
    Matrix2x2 dhdr;
    getdhdr(dhdr,rij,rik);

    //finally, compute all of the forces
    //pnm1 is rij, pn1 is rik
    Dscalar2 vlast,vcur,vnext,vother;
    vcur = d_vc[n_idx(nn,pidx)];
    Dscalar4 vvv = d_vln[n_idx(nn,pidx)];
    vlast.x = vvv.x; vlast.y = vvv.y;
    vnext.x = vvv.z; vnext.y = vvv.z;
    Circumcenter(rij,rik,pno,vother);


    Dscalar2 dAdv,dPdv;
    Dscalar2 dEdv;
    Dscalar  Adiff, Pdiff;
    Dscalar2 dlast, dnext,dcl,dnc;
    Dscalar  dlnorm,dnnorm,dclnorm,dncnorm;

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
    Adiff = KA*(d_AP[pidx].x - d_APpref[pidx].x);
    Pdiff = KA*(d_AP[pidx].y - d_APpref[pidx].y);

    dEdv.x  = 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x;
    dEdv.y  = 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y;

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
    Adiff = KA*(d_AP[neighs.z].x - d_APpref[neighs.z].x);
    Pdiff = KA*(d_AP[neighs.z].y - d_APpref[neighs.z].y);

    dEdv.x  += 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x;
    dEdv.y  += 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y;

    //...and then j
    dAdv.x = 0.5*(vother.y-vlast.y);
    dAdv.y = 0.5*(vlast.x-vother.x);
    //dlast is now -(dnext) from the K calculation
    dlast.x = -dnext.x;
    dlast.y = -dnext.y;
    dlnorm = dnnorm;
    dPdv.x = dlast.x/dlnorm - dcl.x/dclnorm;
    dPdv.y = dlast.y/dlnorm - dcl.y/dclnorm;
    Adiff = KA*(d_AP[neighs.y].x - d_APpref[neighs.y].x);
    Pdiff = KA*(d_AP[neighs.y].y - d_APpref[neighs.y].y);

    dEdv.x  += 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x;
    dEdv.y  += 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y;

    d_forceSets[n_idx(nn,pidx)] = dEdv*dhdr;

    return;
    };

__global__ void gpu_force_sets_tensions_kernel(const Dscalar2* __restrict__ d_points,
                                          const Dscalar2* __restrict__ d_AP,
                                          const Dscalar2* __restrict__ d_APpref,
                                          const int4* __restrict__ d_delSets,
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

    //Great...access the four Delaunay neighbors and the relevant fifth point
    Dscalar2 pi   = d_points[pidx];

    int4 neighs = d_delSets[n_idx(nn,pidx)];
    int neighOther = d_delOther[n_idx(nn,pidx)];
    Dscalar2 pnm2,rij, rik,pn2,pno;

    Box.minDist(d_points[neighs.x],pi,pnm2);
    Box.minDist(d_points[neighs.y],pi,rij);
    Box.minDist(d_points[neighs.z],pi,rik);
    Box.minDist(d_points[neighs.w],pi,pn2);
    Box.minDist(d_points[neighOther],pi,pno);

    //first, compute the derivative of the main voro point w/r/t pidx's position
    Matrix2x2 dhdr;
    getdhdr(dhdr,rij,rik);

    //finally, compute all of the forces
    //pnm1 is rij, pn1 is rik
    Dscalar2 vlast,vcur,vnext,vother;
    vcur = d_vc[n_idx(nn,pidx)];
    Dscalar4 vvv = d_vln[n_idx(nn,pidx)];
    vlast.x = vvv.x; vlast.y = vvv.y;
    vnext.x = vvv.z; vnext.y = vvv.z;
    Circumcenter(rij,rik,pno,vother);


    Dscalar2 dAdv,dPdv,dTdv;
    Dscalar2 dEdv;
    Dscalar  Adiff, Pdiff;
    Dscalar2 dlast, dnext,dcl,dnc;
    Dscalar  dlnorm,dnnorm,dclnorm,dncnorm;
    bool Tik = false;
    bool Tij = false;
    bool Tjk = false;
    if (d_cellTypes[pidx] != d_cellTypes[neighs.z]) Tik = true;
    if (d_cellTypes[pidx] != d_cellTypes[neighs.y]) Tij = true;
    if (d_cellTypes[neighs.z] != d_cellTypes[neighs.y]) Tjk = true;
    //neighs.z is "baseNeigh" of cpu routing... neighs.y is "otherNeigh"....neighOther is "DT_other_idx"

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
    Pdiff = KA*(d_AP[pidx].y - d_APpref[pidx].y);

    dEdv.x  = 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x + gamma*dTdv.x;
    dEdv.y  = 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y + gamma*dTdv.y;

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
    Adiff = KA*(d_AP[neighs.z].x - d_APpref[neighs.z].x);
    Pdiff = KA*(d_AP[neighs.z].y - d_APpref[neighs.z].y);
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

    dEdv.x  += 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x + gamma*dTdv.x;
    dEdv.y  += 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y + gamma*dTdv.y;

    //...and then j
    dAdv.x = 0.5*(vother.y-vlast.y);
    dAdv.y = 0.5*(vlast.x-vother.x);
    dlast.x = -dnext.x;
    dlast.y = -dnext.y;
    dlnorm = dnnorm;
    dPdv.x = dlast.x/dlnorm - dcl.x/dclnorm;
    dPdv.y = dlast.y/dlnorm - dcl.y/dclnorm;
    Adiff = KA*(d_AP[neighs.y].x - d_APpref[neighs.y].x);
    Pdiff = KA*(d_AP[neighs.y].y - d_APpref[neighs.y].y);
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

    dEdv.x  += 2.0*Adiff*dAdv.x +2.0*Pdiff*dPdv.x + gamma*dTdv.x;
    dEdv.y  += 2.0*Adiff*dAdv.y +2.0*Pdiff*dPdv.y + gamma*dTdv.y;

    d_forceSets[n_idx(nn,pidx)] = dEdv*dhdr;

    return;
    };




__global__ void gpu_compute_geometry_kernel(const Dscalar2* __restrict__ d_points,
                                          Dscalar2* __restrict__ d_AP,
                                          const int* __restrict__ d_nn,
                                          const int* __restrict__ d_n,
                                          Dscalar2* __restrict__ d_vc,
                                          Dscalar4* __restrict__ d_vln,
                                          int N,
                                          Index2D n_idx,
                                          gpubox Box
                                        )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    Dscalar2 circumcenter, nnextp, nlastp,pi,rij,rik,vlast,vnext,vfirst;

    int neigh = d_nn[idx];
    Dscalar Varea = 0.0;
    Dscalar Vperi= 0.0;

    pi = d_points[idx];
    nlastp = d_points[ d_n[n_idx(neigh-1,idx)] ];
    nnextp = d_points[ d_n[n_idx(0,idx)] ];

    Box.minDist(nlastp,pi,rij);
    Box.minDist(nnextp,pi,rik);
    Circumcenter(rij,rik,circumcenter);
    vfirst = circumcenter;
    vlast = circumcenter;

    //set the VoroCur to this voronoi vertex
    d_vc[n_idx(0,idx)] = vlast;
    //this vertex is also the "next" vertex of (neighs-1,idx)
    d_vln[n_idx(neigh-1,idx)].z =vlast.x;
    d_vln[n_idx(neigh-1,idx)].w =vlast.y;
    //...and the "last"  vertex of (1,idx)
    d_vln[n_idx(1,idx)].x =vlast.x;
    d_vln[n_idx(1,idx)].y =vlast.y;

    for (int nn = 1; nn < neigh; ++nn)
        {
        rij = rik;
        int nid = d_n[n_idx(nn,idx)];
        nnextp = d_points[ nid ];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(rij,rik,circumcenter);
        vnext = circumcenter;

        //fill in the VoroCur and VoroLastNext structures
        int idl = n_idx(nn-1,idx);
        int idn = n_idx(nn+1,idx);
        if(nn == neigh-1)
            idn = n_idx(0,idx);
        d_vc[n_idx(nn,idx)]=vnext;
        d_vln[idl].z=vnext.x;
        d_vln[idl].w=vnext.y;
        d_vln[idn].x=vnext.x;
        d_vln[idn].y=vnext.y;

        //...and back to computing the geometry
        Varea += TriangleArea(vlast,vnext);
        Dscalar dx = vlast.x - vnext.x;
        Dscalar dy = vlast.y - vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        vlast=vnext;
        };
    Varea += TriangleArea(vlast,vfirst);
    Dscalar dx = vlast.x - vfirst.x;
    Dscalar dy = vlast.y - vfirst.y;
    Vperi += sqrt(dx*dx+dy*dy);

    d_AP[idx].x=Varea;
    d_AP[idx].y=Vperi;

    return;
    };



__global__ void gpu_displace_and_rotate_kernel(Dscalar2 *d_points,
                                          Dscalar2 *d_force,
                                          Dscalar *d_directors,
                                          Dscalar2 *d_motility,
                                          int N,
                                          Dscalar dt,
                                          int seed,
                                          curandState *states,
                                          gpubox Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    curandState_t randState;

    randState=states[idx];
    Dscalar dirx = cosf(d_directors[idx]);
    Dscalar diry = sinf(d_directors[idx]);
    Dscalar v0 = d_motility[idx].x;
    Dscalar Dr = d_motility[idx].y;
    Dscalar angleDiff = cur_norm(&randState)*sqrt(2.0*dt*Dr);
    d_directors[idx] += angleDiff;

    states[idx] = randState;

    d_points[idx].x += dt*(v0*dirx + d_force[idx].x);
    d_points[idx].y += dt*(v0*diry + d_force[idx].y);
    Box.putInBoxReal(d_points[idx]);
    return;
    };


////////////////
//kernel callers
////////////////



bool gpu_init_curand(curandState *states,
                    int N,
                    int Timestep)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    init_curand_kernel<<<nblocks,block_size>>>(states,N,Timestep);
    return cudaSuccess;
    };


bool gpu_compute_geometry(Dscalar2 *d_points,
                        Dscalar2   *d_AP,
                        int      *d_nn,
                        int      *d_n,
                        Dscalar2 *d_vc,
                        Dscalar4 *d_vln,
                        int      N,
                        Index2D  &n_idx,
                        gpubox &Box
                        )
    {
    cudaError_t code;
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_compute_geometry_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_AP,
                                                d_nn,
                                                d_n,
                                                d_vc,
                                                d_vln,
                                                N,
                                                n_idx,
                                                Box
                                                );

    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("compute geometry GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };


bool gpu_displace_and_rotate(Dscalar2 *d_points,
                        Dscalar2 *d_force,
                        Dscalar  *d_directors,
                        Dscalar2 *d_motility,
                        int N,
                        Dscalar dt,
                        int timestep,
                        curandState *states,
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
                                                d_motility,
                                                N,
                                                dt,
                                                timestep,
                                                states,
                                                Box
                                                );
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("displaceAndRotate GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };

bool gpu_force_sets(Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int4   *d_delSets,
                    int    *d_delOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2   *d_nidx,
                    Dscalar  KA,
                    Dscalar  KP,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    )
    {
    cudaError_t code;

    unsigned int block_size = 128;
    if (NeighIdxNum < 128) block_size = 32;
    unsigned int nblocks  = NeighIdxNum/block_size + 1;

    gpu_force_sets_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_AP,
                                                d_APpref,
                                                d_delSets,
                                                d_delOther,
                                                d_vc,
                                                d_vln,
                                                d_forceSets,
                                                d_nidx,
                                                KA,
                                                KP,
                                                NeighIdxNum,
                                                n_idx,
                                                Box
                                                );
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("forceSets GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };




bool gpu_force_sets_tensions(Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int4   *d_delSets,
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
    cudaError_t code;

    unsigned int block_size = 128;
    if (NeighIdxNum < 128) block_size = 32;
    unsigned int nblocks  = NeighIdxNum/block_size + 1;

    gpu_force_sets_tensions_kernel<<<nblocks,block_size>>>(
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
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("forceSets_Tensions GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };

bool gpu_sum_force_sets(
                        Dscalar2 *d_forceSets,
                        Dscalar2 *d_forces,
                        int    *d_nn,
                        int     N,
                        Index2D &n_idx
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_sum_forces_kernel<<<nblocks,block_size>>>(
                                            d_forceSets,
                                            d_forces,
                                            d_nn,
                                            N,
                                            n_idx
            );
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("force_sum GPUassert: %s \n", cudaGetErrorString(code));
    return cudaSuccess;
    };


bool gpu_sum_force_sets_with_exclusions(
                        Dscalar2 *d_forceSets,
                        Dscalar2 *d_forces,
                        Dscalar2 *d_external_forces,
                        int    *d_exes,
                        int    *d_nn,
                        int     N,
                        Index2D &n_idx
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_sum_forces_with_exclusions_kernel<<<nblocks,block_size>>>(
                                            d_forceSets,
                                            d_forces,
                                            d_external_forces,
                                            d_exes,
                                            d_nn,
                                            N,
                                            n_idx
            );
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("force_sum_with_exclusions GPUassert: %s \n", cudaGetErrorString(code));
    return cudaSuccess;
    };


#endif
