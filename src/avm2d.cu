#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "avm2d.cuh"


//!initialize each thread with a different sequence of the same seed of a cudaRNG
__global__ void initialize_curand_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };


//!compute the voronoi vertices for each cell, along with its area and perimeter
__global__ void avm_geometry_kernel(const Dscalar2* __restrict__ d_p,
                                    const Dscalar2* __restrict__ d_v,
                                    const      int* __restrict__ d_nn,
                                    const      int* __restrict__ d_n,
                                    const      int* __restrict__ d_vcn,
                                          Dscalar2*  d_vc,
                                          Dscalar4*  d_vln,
                                          Dscalar2* __restrict__ d_AP,
                                          int N,
                                          Index2D n_idx,
                                          gpubox Box
                                        )
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int neighs = d_nn[idx];
    Dscalar2 cellPos = d_p[idx];
    Dscalar2 vlast, vcur,vnext;
    Dscalar Varea = 0.0;
    Dscalar Vperi = 0.0;

    int vidx = d_n[n_idx(neighs-2,idx)];
    Box.minDist(d_v[vidx],cellPos,vlast);
    vidx = d_n[n_idx(neighs-1,idx)];
    Box.minDist(d_v[vidx],cellPos,vcur);
    for (int nn = 0; nn < neighs; ++nn)
        {
        //for easy force calculation, save the current, last, and next voronoi vertex position in the approprate spot.
        int forceSetIdx = -1;
        for (int ff = 0; ff < 3; ++ff)
            {
           if(d_vcn[3*vidx+ff]==idx)
                forceSetIdx = 3*vidx+ff;
            };

        vidx = d_n[n_idx(nn,idx)];
        Box.minDist(d_v[vidx],cellPos,vnext);
        
        //compute area contribution
        Varea += TriangleArea(vcur,vnext);
        Dscalar dx = vcur.x-vnext.x;
        Dscalar dy = vcur.y-vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        //save voronoi positions in a convenient form
        d_vc[forceSetIdx] = vcur;
        d_vln[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
        //advance the loop
        vlast = vcur;
        vcur = vnext;
        };
    d_AP[idx].x=Varea;
    d_AP[idx].y=Vperi;
    };

//!compute the force on a vertex due to one of the three cells
__global__ void avm_force_sets_kernel(
                        int      *d_vcn,
                        Dscalar2 *d_vc,
                        Dscalar4 *d_vln,
                        Dscalar2 *d_AP,
                        Dscalar2 *d_APpref,
                        Dscalar2 *d_fs,
                        int nForceSets,
                        Dscalar KA, Dscalar KP)
    {
    // read in the cell index that belongs to this thread
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;

    Dscalar2 vlast,vnext;

    int cellIdx = d_vcn[fsidx];
    Dscalar Adiff = KA*(d_AP[cellIdx].x - d_APpref[cellIdx].x);
    Dscalar Pdiff = KP*(d_AP[cellIdx].y - d_APpref[cellIdx].y);

    //vcur = d_vc[fsidx];
    vlast.x = d_vln[fsidx].x;
    vlast.y = d_vln[fsidx].y;
    vnext.x = d_vln[fsidx].z;
    vnext.y = d_vln[fsidx].w;
    computeForceSetAVM(d_vc[fsidx],vlast,vnext,Adiff,Pdiff,d_fs[fsidx]);
    };



//!sum up the force sets to get the force on each vertex
__global__ void avm_sum_force_sets_kernel(
                                    const Dscalar2* __restrict__ d_fs,
                                    Dscalar2* __restrict__ d_f,
                                    int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    Dscalar2 ftemp;
    ftemp.x = 0.0; ftemp.y=0.0;
    for (int ff = 0; ff < 3; ++ff)
        {
        ftemp.x += d_fs[3*idx+ff].x;
        ftemp.y += d_fs[3*idx+ff].y;
        };
    d_f[idx] = ftemp;
    };

//!sum up the force sets to get the force on each vertex
__global__ void avm_displace_vertices_kernel(
                                        Dscalar2 *d_v,
                                        Dscalar2 *d_f,
                                        Dscalar  *d_cd,
                                        int      *d_vcn,
                                        Dscalar  v0,
                                        Dscalar  deltaT,
                                        gpubox   Box,
                                        int      Nvertices)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nvertices)
        return;

    //the vertex motility is the average of th motility of the connected cells
    Dscalar directorx = (Cos(d_cd[d_vcn[3*idx]])+Cos(d_cd[d_vcn[3*idx+1]])+Cos(d_cd[d_vcn[3*idx+2]]))/3.0;
    Dscalar directory = (Sin(d_cd[d_vcn[3*idx]])+Sin(d_cd[d_vcn[3*idx+1]])+Sin(d_cd[d_vcn[3*idx+2]]))/3.0;
    //update positions from forces and motility
    d_v[idx].x += deltaT*(v0*directorx + d_f[idx].x);
    d_v[idx].y += deltaT*(v0*directory + d_f[idx].y);
    //make sure the vertices stay in the box
    Box.putInBoxReal(d_v[idx]);
    };

//!sum up the force sets to get the force on each vertex
__global__ void avm_rotate_directors_kernel(
                                        Dscalar  *d_cd,
                                        curandState *d_cs,
                                        Dscalar  Dr,
                                        Dscalar  deltaT,
                                        int      Ncells)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;

    //get the per-cell RNG, rotate the director, return the RNG
    curandState_t randState;
    randState=d_cs[idx];
    d_cd[idx] += cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    d_cs[idx] = randState;
    };


//!compute the average position of the vertices of each cell, store as the "cell position"
__global__ void avm_get_cell_positions_kernel(Dscalar2* d_p,
                                              Dscalar2* d_v,
                                              int    * d_nn,
                                              int    * d_n,
                                              int N,
                                              Index2D n_idx,
                                              gpubox Box)
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    Dscalar2 vertex, oldCellPos,pos;
    pos.x=0.0;pos.y=0.0;
    oldCellPos=d_p[idx];
    int neighs = d_nn[idx];
    for (int n = 0; n < neighs; ++n)
        {
        Box.minDist(d_v[ d_n[n_idx(n,idx)] ],oldCellPos,vertex);
        pos.x += vertex.x;
        pos.y += vertex.y;
        };
    pos.x /= neighs;
    pos.y /= neighs;
    Box.putInBoxReal(pos);
    d_p[idx] = pos;
    };






//!Call the kernel to initialize a different RNG for each particle
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    initialize_curand_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_avm_geometry(
                    Dscalar2 *d_p,
                    Dscalar2 *d_v,
                    int      *d_nn,
                    int      *d_n,
                    int      *d_vcn,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_AP,
                    int      N, 
                    Index2D  &n_idx, 
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    avm_geometry_kernel<<<nblocks,block_size>>>(d_p,d_v,d_nn,d_n,d_vcn,d_vc,d_vln,d_AP,N, n_idx, Box);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to calculate force sets
bool gpu_avm_force_sets(
                    int      *d_vcn,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    Dscalar2 *d_fs,
                    int nForceSets,
                    Dscalar KA, Dscalar KP)
    {
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks  = nForceSets/block_size + 1;

    avm_force_sets_kernel<<<nblocks,block_size>>>(d_vcn,d_vc,d_vln,d_AP,d_APpref,d_fs,nForceSets,KA,KP);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to sum up the force sets to get net force on each vertex
bool gpu_avm_sum_force_sets(
                    Dscalar2 *d_fs,
                    Dscalar2 *d_f,
                    int      Nvertices)
    {
    unsigned int block_size = 128;
    if (Nvertices < 128) block_size = 32;
    unsigned int nblocks  = Nvertices/block_size + 1;


    avm_sum_force_sets_kernel<<<nblocks,block_size>>>(d_fs,d_f,Nvertices);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_avm_displace_and_rotate(
                    Dscalar2 *d_v,
                    Dscalar2 *d_f,
                    Dscalar  *d_cd,
                    int      *d_vcn,
                    curandState *d_cs,
                    Dscalar  v0,
                    Dscalar  Dr,
                    Dscalar  deltaT,
                    gpubox   &Box,
                    int      Nvertices,
                    int      Ncells)
    {
    unsigned int block_size = 128;
    if (Nvertices < 128) block_size = 32;
    unsigned int nblocks  = Nvertices/block_size + 1;

    //displace vertices
    avm_displace_vertices_kernel<<<nblocks,block_size>>>(d_v,d_f,d_cd,d_vcn,v0,deltaT,Box,Nvertices);
    cudaThreadSynchronize();
    //rotate cell directors
    if (Ncells < 128) block_size = 32;
    nblocks = Ncells/block_size + 1;
    avm_rotate_directors_kernel<<<nblocks,block_size>>>(d_cd,d_cs,Dr,deltaT,Ncells);
    cudaThreadSynchronize();

    return cudaSuccess;
    };

//!Call the kernel to calculate the position of each cell from the position of its vertices
bool gpu_avm_get_cell_positions(
                    Dscalar2 *d_p,
                    Dscalar2 *d_v,
                    int      *d_nn,
                    int      *d_n,
                    int      N, 
                    Index2D  &n_idx, 
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    avm_get_cell_positions_kernel<<<nblocks,block_size>>>(d_p,d_v,d_nn,d_n,N, n_idx, Box);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

