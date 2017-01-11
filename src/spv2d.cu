#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "cellListGPU.cuh"
#include "spv2d.cuh"

#include "indexer.h"
#include "gpubox.h"
#include "cu_functions.h"
#include <iostream>
#include <stdio.h>
#include "Matrix.h"

/*!
A file defining some global kernels for use in the spv2d class
*/

/*!
    \addtogroup spvKernels
    @{
*/

/*! 
  Each thread -- corresponding to each Voronoi cell -- is initialized with a different sequence
  of the same seed of a cudaRNG
*/
__global__ void init_curand_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };


/*!
  Each cell has a force contribution due to the derivative of the energy with respect to each of
  its voronoi vertices... add them up to get the force per cell.
  */
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

/*!
  add up force sets, as above, but keep track of exclusions
  */
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

/*!
  the force on a particle is decomposable into the force contribution from each of its voronoi
  vertices...calculate those sets of forces
  */
__global__ void gpu_force_sets_kernel(const Dscalar2* __restrict__ d_points,
                                      const Dscalar2* __restrict__ d_AP,
                                      const Dscalar2*  __restrict__ d_APpref,
                                      const int2* __restrict__ d_delSets,
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
    int nidx=n_idx(nn,pidx);

    //local variables declared...
    Dscalar2 dAdv,dPdv;
    Dscalar2 dEdv;
    Dscalar  Adiff, Pdiff;
    Dscalar2 dlast, dnext,dcl,dnc;
    Dscalar  dlnorm,dnnorm,dclnorm,dncnorm;
    Dscalar2 vlast,vcur,vnext,vother;

    //logically, I want these variables:
    //Dscalar2 pi, rij, rik,pno;
    //they will simply re-use
    //     dlast, dnext, dcl, dnc, respectively
    //to reduce register usage


    //Great...access the Delaunay neighbors and the relevant other point
    int2 neighs;
    dlast   = d_points[pidx];

    neighs = d_delSets[nidx];

    Box.minDist(d_points[neighs.x],dlast,dnext);
    Box.minDist(d_points[neighs.y],dlast,dcl);
    Box.minDist(d_points[d_delOther[nidx]],dlast,dnc);

    //first, compute the derivative of the main voro point w/r/t pidx's position
    Matrix2x2 dhdr;
    getdhdr(dhdr,dnext,dcl);

    //finally, compute all of the forces
    //pnm1 is rij (dnext), pn1 is rik
    vcur = d_vc[nidx];
    Dscalar4 vvv = d_vln[nidx];
    vlast.x = vvv.x; vlast.y = vvv.y;
    vnext.x = vvv.z; vnext.y = vvv.w;

    Circumcenter(dnext,dcl,dnc,vother);


    //self terms
    dAdv.x = 0.5*(vlast.y-vnext.y);
    dAdv.y = 0.5*(vnext.x-vlast.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y=vlast.y-vcur.y;
    dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
#ifdef SCALARFLOAT
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
    if(dlnorm < THRESHOLD)
        dlnorm = THRESHOLD;
#endif
    //save a few of these differences for later...
    //dcl.x = -dlast.x;dcl.y = -dlast.y;
    //dnc.x=-dnext.x;dnc.y=-dnext.y;
    dcl.x = dlast.x; dcl.y = dlast.y;
    dnc.x = dnext.x; dnc.y = dnext.y;
    dclnorm=dlnorm;
    dncnorm=dnnorm;

    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
    Adiff = KA*(d_AP[pidx].x - d_APpref[pidx].x);
    Pdiff = KP*(d_AP[pidx].y - d_APpref[pidx].y);

    //replace all "multiply-by-two's" with a single one at the end...saves 10 mult operations
    dEdv.x  = Adiff*dAdv.x + Pdiff*dPdv.x;
    dEdv.y  = Adiff*dAdv.y + Pdiff*dPdv.y;

    //other terms...k first...
    dAdv.x = 0.5*(vnext.y-vother.y);
    dAdv.y = 0.5*(vother.x-vnext.x);
    dnext.x = vcur.x-vother.x;
    dnext.y = vcur.y-vother.y;
    dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
#ifdef SCALARFLOAT
    if(dnnorm < THRESHOLD)
        dnnorm = THRESHOLD;
#endif
    dPdv.x = -dnc.x/dncnorm - dnext.x/dnnorm;
    dPdv.y = -dnc.y/dncnorm - dnext.y/dnnorm;
    Adiff = KA*(d_AP[neighs.y].x - d_APpref[neighs.y].x);
    Pdiff = KP*(d_AP[neighs.y].y - d_APpref[neighs.y].y);

    dEdv.x  += Adiff*dAdv.x + Pdiff*dPdv.x;
    dEdv.y  += Adiff*dAdv.y + Pdiff*dPdv.y;

    //...and then j
    dAdv.x = 0.5*(vother.y-vlast.y);
    dAdv.y = 0.5*(vlast.x-vother.x);
    //dlast is now -(dnext) from the K calculation
    //dlast.x = -dnext.x;
    //dlast.y = -dnext.y;
    //dlnorm = dnnorm;
    dPdv.x = -dnext.x/dnnorm + dcl.x/dclnorm;
    dPdv.y = -dnext.y/dnnorm + dcl.y/dclnorm;
    Adiff = KA*(d_AP[neighs.x].x - d_APpref[neighs.x].x);
    Pdiff = KP*(d_AP[neighs.x].y - d_APpref[neighs.x].y);

    dEdv.x  += Adiff*dAdv.x + Pdiff*dPdv.x;
    dEdv.y  += Adiff*dAdv.y + Pdiff*dPdv.y;

    dEdv.x *= 2.0;
    dEdv.y *= 2.0;

    d_forceSets[nidx] = dEdv*dhdr;

    return;
    };

/*!
The force on a particle is decomposable into the force contribution from each of its voronoi vertices
...calculate those sets of forces with an additional tension term between cells of different type
*/
__global__ void gpu_force_sets_tensions_kernel(const Dscalar2* __restrict__ d_points,
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



/*!
  Since the cells are guaranteed to be convex, the area of the cell is the sum of the areas of
  the triangles formed by consecutive Voronoi vertices
  */
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

    Dscalar2  nnextp, nlastp,pi,rij,rik,vlast,vnext,vfirst;

    int neigh = d_nn[idx];
    Dscalar Varea = 0.0;
    Dscalar Vperi= 0.0;

    pi = d_points[idx];
    nlastp = d_points[ d_n[n_idx(neigh-1,idx)] ];
    nnextp = d_points[ d_n[n_idx(0,idx)] ];

    Box.minDist(nlastp,pi,rij);
    Box.minDist(nnextp,pi,rik);
    Circumcenter(rij,rik,vfirst);
    vlast = vfirst;

    //set the VoroCur to this voronoi vertex
    //the convention is that nn=0 in this routine should be nn = 1 in the force sets calculation
    d_vc[n_idx(1,idx)] = vlast;

    for (int nn = 1; nn < neigh; ++nn)
        {
        rij = rik;
        int nid = d_n[n_idx(nn,idx)];
        nnextp = d_points[ nid ];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(rij,rik,vnext);

        //fill in the VoroCur structure

        int idc = n_idx(nn+1,idx);
        if(nn == neigh-1)
            idc = n_idx(0,idx);

        d_vc[idc]=vnext;

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

    //it's more memory-access friendly to now fill in the VoroLastNext structure separately
    vlast = d_vc[n_idx(neigh-1,idx)];
    vfirst = d_vc[n_idx(0,idx)];
    for (int nn = 0; nn < neigh; ++nn)
        {
        int idn = n_idx(nn+1,idx);
        if(nn == neigh-1) idn = n_idx(0,idx);
        vnext = d_vc[idn];

        int idc = n_idx(nn,idx);
        d_vln[idc].x = vlast.x;
        d_vln[idc].y = vlast.y;
        d_vln[idc].z = vnext.x;
        d_vln[idc].w = vnext.y;

        vlast = vfirst;
        vfirst = vnext;
        };

    d_AP[idx].x=Varea;
    d_AP[idx].y=Vperi;

    return;
    };


/*!
  move particles according to their motility and forces...each thread checks out its RNG, advances
  it by one increment, and returns it.
  */
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
    Dscalar dirx = Cos(d_directors[idx]);
    Dscalar diry = Sin(d_directors[idx]);
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

//!Call the kernel to initialize a different RNG for each particle
bool gpu_init_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    init_curand_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    HANDLE_ERROR(cudaGetLastError());
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to compute the geometry
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
    //cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to move particles around
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
    //cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to compute the force sets
bool gpu_force_sets(Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
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
    HANDLE_ERROR(cudaGetLastError());
    //cudaThreadSynchronize();
    return cudaSuccess;
    };



//!Call the kernel to compute force sets with additional tension terms
bool gpu_force_sets_tensions(Dscalar2 *d_points,
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
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!call the kernel to add up the forces
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
    //cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


//!call the kernel to add up forces with particle exclusions
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
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
