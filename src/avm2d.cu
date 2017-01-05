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
                                    const      int* __restrict__ d_cvn,
                                    const      int* __restrict__ d_cv,
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
    int neighs = d_cvn[idx];
    Dscalar2 cellPos = d_p[idx];
    Dscalar2 vlast, vcur,vnext;
    Dscalar Varea = 0.0;
    Dscalar Vperi = 0.0;

    int vidx = d_cv[n_idx(neighs-2,idx)];
    Box.minDist(d_v[vidx],cellPos,vlast);
    vidx = d_cv[n_idx(neighs-1,idx)];
    Box.minDist(d_v[vidx],cellPos,vcur);
if (d_cvn[n_idx(d_cvn[idx]-1, idx)] ==d_cvn[n_idx(d_cvn[idx]-2,idx)])
printf("cell double%i\n",idx);
    for (int nn = 0; nn < neighs; ++nn)
        {
        //for easy force calculation, save the current, last, and next voronoi vertex position in the approprate spot.
        int forceSetIdx = -1;
        for (int ff = 0; ff < 3; ++ff)
            {
           if(d_vcn[3*vidx+ff]==idx)
                forceSetIdx = 3*vidx+ff;
            };

if (forceSetIdx <0 || forceSetIdx >= 6*N || idx == 171)
{
printf("forceSetIdx = %i\t vidx = %i\t nn=%i\n",forceSetIdx,vidx,nn);
printf("cell = %i;  vidx is connected to:",idx);
for (int ff = 0; ff < 3; ++ff)
    printf("%i, ",d_vcn[3*vidx+ff]);
printf("\ncell is connected to:");
int cneigh = d_cvn[idx];
for (int ff = 0; ff < cneigh; ++ff)
    printf("%i, ",d_cvn[n_idx(ff,idx)]);
printf("\n");
};
        vidx = d_cv[n_idx(nn,idx)];
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


//    printf("cell %f\t %f\n",deltaT*(v0*directorx), deltaT*d_f[idx].x);


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

/*!
Because operations are performed in parallel, the GPU routine will break if the same vertex
is involved in multiple T1 transitions in the same time step. Defend against that by limiting
the number of flips to one.
*/
__global__ void avm_defend_against_multiple_T1_kernel(
                                        int *d_flip,
                                        int *d_vn,
                                        int Nvertices)
    {
    unsigned int vertex1 = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex1 >= Nvertices)
        return;
    //if the first vertex-neighbor is to be flipped, prevent any other nearby flips
    if (d_flip[3*vertex1] == 1)
        {
        for (int ff = 0; ff < 3; ++ff)
            {
            int vertex2 = d_vn[3*vertex1+ff];
            for(int f2=0;f2 <3; ++f2)
                d_flip[3*vertex2+f2]=0;
            };
        d_flip[3*vertex1+1] = 0;
        d_flip[3*vertex1+2] = 0;
        };

    //if the second vertex-neighbor is to be flipped, prevent any other flips of the two vertices
    if (d_flip[3*vertex1+1] == 1)
        {
        for (int ff = 0; ff < 3; ++ff)
            {
            int vertex2 = d_vn[3*vertex1+ff];
            for(int f2=0;f2 <3; ++f2)
                d_flip[3*vertex2+f2]=0;
            };
        d_flip[3*vertex1+2] = 0;
        };

    //if the third vertex-neighbor is to be flipped, prevent any other flips of the two vertices
    if (d_flip[3*vertex1+2] == 1)
        {
        for (int ff = 0; ff < 3; ++ff)
            {
            int vertex2 = d_vn[3*vertex1+ff];
            for(int f2=0;f2 <3; ++f2)
                d_flip[3*vertex2+f2]=0;
            };
        };
    };


//!Run through every pair of vertices (once), see if any T1 transitions should be done, and see if the cell-vertex list needs to grow
__global__ void avm_simple_T1_test_kernel(Dscalar2* d_v,
                                        int      *d_vn,
                                        int      *d_vflip,
                                        int      *d_vcn,
                                        int      *d_cvn,
                                        gpubox   Box,
                                        Dscalar  T1THRESHOLD,
                                        int      NvTimes3,
                                        int      vertexMax,
                                        int      *d_grow)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= NvTimes3)
        return;
    int vertex1 = idx/3;
    int vertex2 = d_vn[idx];
    Dscalar2 edge;
    if(vertex1 < vertex2)
        {
        Box.minDist(d_v[vertex1],d_v[vertex2],edge);
        if(norm(edge) < T1THRESHOLD)
            {
            d_vflip[idx]=1;
            //test the number of neighbors of the cells connected to v1 and v2 to see if the cell list should grow
            //this is kind of slow, and I wish I could optimize it away, or at least not test for it during
            //every time step. The latter seems pretty doable.
            if(d_cvn[d_vcn[3*vertex1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cvn[d_vcn[3*vertex1+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cvn[d_vcn[3*vertex1+2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cvn[d_vcn[3*vertex2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cvn[d_vcn[3*vertex2+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cvn[d_vcn[3*vertex2+2]] == vertexMax)
                d_grow[0] = 1;
            }
        else
            d_vflip[idx]=0;
        }
    else
        d_vflip[idx] = 0;

    };

//!flip any edge label for re-wiring
__global__ void avm_flip_edges_kernel(int* d_vflip,
                                      Dscalar2 *d_v,
                                      int      *d_vn,
                                      int      *d_vcn,
                                      int      *d_cvn,
                                      int      *d_cv,
                                      gpubox   Box,
                                      Index2D  n_idx,
                                      int      NvTimes3)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //return if the index is out of bounds or if the edge isn't marked for flipping
    if (idx >= NvTimes3 || d_vflip[idx] == 0)
        return;
    //identify the vertices and reset the flag
    int vertex1 = idx/3;
    int vertex2 = d_vn[idx];
    d_vflip[idx] = 0;

    printf("T1 for vertices %i %i ...\n",vertex1,vertex2);

    //Rotate the vertices in the edge and set them at twice their original distance
    Dscalar2 edge;
    Dscalar2 v1 = d_v[vertex1];
    Dscalar2 v2 = d_v[vertex2];
    Box.minDist(v1,v2,edge);

    Dscalar2 midpoint;
    midpoint.x = v2.x + 0.5*edge.x;
    midpoint.y = v2.y + 0.5*edge.y;

    v1.x = midpoint.x-edge.y;v1.y = midpoint.y+edge.x;
    v2.x = midpoint.x+edge.y;v2.y = midpoint.y-edge.x;
    Box.putInBoxReal(v1);
    Box.putInBoxReal(v2);
    d_v[vertex1] = v1;
    d_v[vertex2] = v2;

    //now, do the gross work of cell and vertex rewiring
    int4 cellSet;cellSet.x=-1;cellSet.y=-1;cellSet.z=-1;cellSet.w=-1;
    int4 vertexSet;
    ///////////////////////////////////////////////////
    //TERRIBLE GPU CODE = COPY THE CPU BRANCH LOGIC....
    ///////////////////////////////////////////////////
    int cell1,cell2,cell3,ctest;
    int vlast, vcur, vnext, cneigh;
    cell1 = d_vcn[3*vertex1];
    cell2 = d_vcn[3*vertex1+1];
    cell3 = d_vcn[3*vertex1+2];
    //cell_l doesn't contain vertex 1, so it is the cell neighbor of vertex 2 we haven't found yet
    for (int ff = 0; ff < 3; ++ff)
        {
        ctest = d_vcn[3*vertex2+ff];
        if(ctest != cell1 && ctest != cell2 && ctest != cell3)
            cellSet.w=ctest;
        };
    //find vertices "c" and "d"
    cneigh = d_cvn[cellSet.w];
    vlast = d_cv[ n_idx(cneigh-2,cellSet.w) ];
    vcur = d_cv[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cell1)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };

    //classify cell1
    cneigh = d_cvn[cell1];
    vlast = d_cv[ n_idx(cneigh-2,cell1) ];
    vcur = d_cv[ n_idx(cneigh-1,cell1) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cell1)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell1;
    else if(vnext == vertex2)
        cellSet.z = cell1;
    else
        {
        cellSet.y = cell1;
        };

    //classify cell2
    cneigh = d_cvn[cell2];
    vlast = d_cv[ n_idx(cneigh-2,cell2) ];
    vcur = d_cv[ n_idx(cneigh-1,cell2) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cell2)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell2;
    else if(vnext == vertex2)
        cellSet.z = cell2;
    else
        {
        cellSet.y = cell2;
        };

    //classify cell3
    cneigh = d_cvn[cell3];
    vlast = d_cv[ n_idx(cneigh-2,cell3) ];
    vcur = d_cv[ n_idx(cneigh-1,cell3) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cell3)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell3;
    else if(vnext == vertex2)
        cellSet.z = cell3;
    else
        {
        cellSet.y = cell3;
        };

    //get the vertexSet by examining cells j and l
    cneigh = d_cvn[cellSet.y];
    vlast = d_cv[ n_idx(cneigh-2,cellSet.y) ];
    vcur = d_cv[ n_idx(cneigh-1,cellSet.y) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cellSet.y)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.x=vlast;
    vertexSet.y=vnext;
    cneigh = d_cvn[cellSet.w];
    vlast = d_cv[ n_idx(cneigh-2,cellSet.w) ];
    vcur = d_cv[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cv[n_idx(cn,cellSet.w)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.w=vlast;
    vertexSet.z=vnext;
    ///////////////////////////////////////////////////
    //END OF FIRST CHUNK OF TERRIBLE CODE...but the nightmare isn't over
    ///////////////////////////////////////////////////

    //re-wire the cells and vertices
    //start with the vertex-vertex and vertex-cell  neighbors
    for (int vert = 0; vert < 3; ++vert)
        {
        //vertex-cell neighbors
        if(d_vcn[3*vertex1+vert] == cellSet.z)
            d_vcn[3*vertex1+vert] = cellSet.w;
        if(d_vcn[3*vertex2+vert] == cellSet.x)
            d_vcn[3*vertex2+vert] = cellSet.y;
        //vertex-vertex neighbors
        if(d_vn[3*vertexSet.y+vert] == vertex1)
            d_vn[3*vertexSet.y+vert] = vertex2;
        if(d_vn[3*vertexSet.z+vert] == vertex2)
            d_vn[3*vertexSet.z+vert] = vertex1;
        if(d_vn[3*vertex1+vert] == vertexSet.y)
            d_vn[3*vertex1+vert] = vertexSet.z;
        if(d_vn[3*vertex2+vert] == vertexSet.z)
            d_vn[3*vertex2+vert] = vertexSet.y;
        };
    //now rewire the cells
    //cell i loses v2 as a neighbor

//    printf("(%i,%i)\t cells: (%i %i %i %i), vertices: (%i,%i,%i,%i)\n",vertex1,vertex2,cellSet.x,cellSet.y,cellSet.z,cellSet.w,vertexSet.x,vertexSet.y,vertexSet.z,vertexSet.w);
/*
if(cellSet.x<0)
    {
    printf("(%i,%i)\t cells: (%i %i %i %i), vertices: (%i,%i,%i,%i)\n",vertex1,vertex2,cellSet.x,cellSet.y,cellSet.z,cellSet.w,vertexSet.x,vertexSet.y,vertexSet.z,vertexSet.w);
    cneigh = d_cvn[d_vcn[3*vertex1]];
    printf("Cell 1, Cellidx %i:",d_vcn[3*vertex1]);
    for (int c1 = 0; c1 < cneigh; ++c1)
        printf("%i\t",d_cv[n_idx(c1,d_vcn[3*vertex1])] );
    printf("\n");
    cneigh = d_cvn[d_vcn[3*vertex1+1]];
    printf("Cell 2, Cellidx %i:",d_vcn[3*vertex1+1]);
    for (int c1 = 0; c1 < cneigh; ++c1)
        printf("%i\t",d_cv[n_idx(c1,d_vcn[3*vertex1+1])] );
    printf("\n");
    cneigh = d_cvn[d_vcn[3*vertex1+2]];
    printf("Cell 3, Cellidx %i:",d_vcn[3*vertex1+2]);
    for (int c1 = 0; c1 < cneigh; ++c1)
        printf("%i\t",d_cv[n_idx(c1,d_vcn[3*vertex1+2])] );
    printf("\n");
    };
*/
    cneigh = d_cvn[cellSet.x];
    int cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cv[n_idx(cc,cellSet.x)] == vertex2)
            cidx +=1;
        d_cv[n_idx(cc,cellSet.x)] = d_cv[n_idx(cidx,cellSet.x)];
        cidx +=1;
        };
    d_cvn[cellSet.x] -= 1;

    //cell j gains v2 in between v1 and b, so step through list backwards and insert
    cneigh = d_cvn[cellSet.y];
    cidx = cneigh;
    int vLocation = cneigh;
    for (int cc = cneigh-1;cc >=0; --cc)
        {
        int cellIndex = d_cv[n_idx(cc,cellSet.y)];
        if(cellIndex == vertex1)
            {
            vLocation = cidx;
            cidx -= 1;
            };
        d_cv[n_idx(cidx,cellSet.y)] = cellIndex;
        cidx -= 1;
        };
    if(cidx ==0)
        d_cv[n_idx(0,cellSet.y)] = vertex2;
    else
        d_cv[n_idx(vLocation,cellSet.y)] = vertex2;
    d_cvn[cellSet.y] += 1;

    //cell k loses v1 as a neighbor
    cneigh = d_cvn[cellSet.z];
    cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cv[n_idx(cc,cellSet.z)] == vertex1)
            cidx +=1;
        d_cv[n_idx(cc,cellSet.z)] = d_cv[n_idx(cidx,cellSet.z)];
        cidx +=1;
        };
    d_cvn[cellSet.z] -= 1;

    //cell l gains v1 in between v2 and c...copy the logic of cell j
    cneigh = d_cvn[cellSet.w];
    cidx = cneigh;
    vLocation = cneigh;
    for (int cc = cneigh-1;cc >=0; --cc)
        {
        int cellIndex = d_cv[n_idx(cc,cellSet.w)];
        if(cellIndex == vertex2)
            {
            vLocation = cidx;
            cidx -= 1;
            };
        d_cv[n_idx(cidx,cellSet.w)] = cellIndex;
        cidx -= 1;
        };
    if(cidx ==0)
        d_cv[n_idx(0,cellSet.w)] = vertex1;
    else
        d_cv[n_idx(vLocation,cellSet.w)] = vertex1;
    d_cvn[cellSet.w] += 1;

    ///////////////////////////////////////////////////
    //END OF COPIED CODE
    ///////////////////////////////////////////////////
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

    Dscalar2 vertex, pos, baseVertex;
    pos.x=0.0;pos.y=0.0;
    baseVertex = d_v[ d_n[n_idx(0,idx)] ];
    int neighs = d_nn[idx];
    for (int n = 1; n < neighs; ++n)
        {
        Box.minDist(d_v[ d_n[n_idx(n,idx)] ],baseVertex,vertex);
        pos.x += vertex.x;
        pos.y += vertex.y;
        };
    pos.x /= neighs;
    pos.y /= neighs;
    pos.x += baseVertex.x;
    pos.y += baseVertex.y;
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
                    int      *d_cvn,
                    int      *d_cv,
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


    avm_geometry_kernel<<<nblocks,block_size>>>(d_p,d_v,d_cvn,d_cv,d_vcn,d_vc,d_vln,d_AP,N, n_idx, Box);
    cudaThreadSynchronize();
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("compute geometry GPUassert: %s \n", cudaGetErrorString(code));
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
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("compute force sets GPUassert: %s \n", cudaGetErrorString(code));
    cudaThreadSynchronize();
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
    cudaError_t code = cudaGetLastError();
    cudaThreadSynchronize();
    if(code!=cudaSuccess)
        printf("sum force sets GPUassert: %s \n", cudaGetErrorString(code));
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
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("displace and rotate GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };


//!Call the kernel to test every edge for a T1 event, see if vertexMax needs to increase
bool gpu_avm_test_edges_for_T1(
                    Dscalar2 *d_v,
                    int      *d_vn,
                    int      *d_vflip,
                    int      *d_vcn,
                    int      *d_cvn,
                    gpubox   &Box,
                    Dscalar  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow)
    {
    unsigned int block_size = 128;
    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    unsigned int nblocks  = NvTimes3/block_size + 1;

    //test edges
    avm_simple_T1_test_kernel<<<nblocks,block_size>>>(
            d_v,d_vn,d_vflip,d_vcn,d_cvn,
            Box,
            T1THRESHOLD,
            NvTimes3,vertexMax,d_grow);

    cudaThreadSynchronize();
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("test for T1 GPUassert: %s \n", cudaGetErrorString(code));

    //only allow a vertex to be in one T1 transition in a given time step
    if(Nvertices<128) block_size = 32;
    nblocks = Nvertices/block_size + 1;
    avm_defend_against_multiple_T1_kernel<<<nblocks,block_size>>>(
                                        d_vflip,
                                        d_vn,
                                        Nvertices);
    code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("One T1 per vertex per timestep GPUassert: %s \n", cudaGetErrorString(code));

    return cudaSuccess;
    };

//!Call the kernel to test every edge for a T1 event, see if vertexMax needs to increase
bool gpu_avm_flip_edges(
                    int      *d_vflip,
                    Dscalar2 *d_v,
                    int      *d_vn,
                    int      *d_vcn,
                    int      *d_cvn,
                    int      *d_cv,
                    gpubox   &Box,
                    Index2D  &n_idx,
                    int      Nvertices)
    {
    unsigned int block_size = 128;
    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    unsigned int nblocks  = NvTimes3/block_size + 1;


    //test edges
    avm_flip_edges_kernel<<<nblocks,block_size>>>(
            d_vflip,d_v,d_vn,d_vcn,d_cvn,d_cv,
            Box,n_idx,
            NvTimes3);

    cudaThreadSynchronize();
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        printf("flip edges GPUassert: %s \n", cudaGetErrorString(code));
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
    cudaThreadSynchronize();
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("get cell positions GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };
    return cudaSuccess;
    };

