#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "avm2d.cuh"
#include "lock.h"

/** \file avm.cu
    * Defines kernel callers and kernels for GPU calculations of AVM parts
*/

/*!
    \addtogroup avmKernels
    @{
*/

/*!
  Each thread -- corresponding to each Voronoi cell -- is initialized with a different sequence
  of the same seed of a cudaRNG
*/
__global__ void initialize_curand_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };


/*!
  Since the cells are NOT guaranteed to be convex, the area of the cell must take into account any
  self-intersections. The strategy is the same as in the CPU branch.
  */
__global__ void avm_geometry_kernel(
                                   const Dscalar2* __restrict__ d_vertexPositions,
                                   const int*  __restrict__ d_cellVertexNum,
                                   const int*  __restrict__ d_cellVertices,
                                   const int*  __restrict__ d_vertexCellNeighbors,
                                   Dscalar2*  __restrict__ d_voroCur,
                                   Dscalar4*  __restrict__ d_voroLastNext,
                                   Dscalar2*  __restrict__ d_AreaPerimeter,
                                   int N,
                                   Index2D n_idx,
                                   gpubox Box
                                    )
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int neighs = d_cellVertexNum[idx];
    //Define the vertices of a cell relative to some (any one ) of its vertices to take care of periodic BCs
    Dscalar2 cellPos = d_vertexPositions[ d_cellVertices[n_idx(neighs-2,idx)]];
    Dscalar2 vlast, vcur,vnext;
    Dscalar Varea = 0.0;
    Dscalar Vperi = 0.0;

    vlast.x = 0.0; vlast.y=0.0;
    int vidx = d_cellVertices[n_idx(neighs-1,idx)];
    Box.minDist(d_vertexPositions[vidx],cellPos,vcur);
    for (int nn = 0; nn < neighs; ++nn)
        {
        //for easy force calculation, save the current, last, and next voronoi vertex position
        //in the approprate spot.
        int forceSetIdx = -1;
        for (int ff = 0; ff < 3; ++ff)
            {
            if(forceSetIdx != -1) continue;
            if(d_vertexCellNeighbors[3*vidx+ff]==idx)
                forceSetIdx = 3*vidx+ff;
            };

        vidx = d_cellVertices[n_idx(nn,idx)];
        Box.minDist(d_vertexPositions[vidx],cellPos,vnext);

        //compute area contribution. It is
        // 0.5 * (vcur.x+vnext.x)*(vnext.y-vcur.y)
        Varea += SignedPolygonAreaPart(vcur,vnext);
        Dscalar dx = vcur.x-vnext.x;
        Dscalar dy = vcur.y-vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        //save voronoi positions in a convenient form
        d_voroCur[forceSetIdx] = vcur;
        d_voroLastNext[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
        //advance the loop
        vlast = vcur;
        vcur = vnext;
        };
    d_AreaPerimeter[idx].x=Varea;
    d_AreaPerimeter[idx].y=Vperi;
    };

/*!
  The force on a vertex has a contribution from how moving that vertex affects each of the neighboring
cells...compute those force sets
*/
__global__ void avm_force_sets_kernel(
                        int      *d_vertexCellNeighbors,
                        Dscalar2 *d_voroCur,
                        Dscalar4 *d_voroLastNext,
                        Dscalar2 *d_AreaPerimeter,
                        Dscalar2 *d_AreaPerimeterPreferences,
                        Dscalar2 *d_vertexForceSets,
                        int nForceSets,
                        Dscalar KA, Dscalar KP)
    {
    // read in the cell index that belongs to this thread
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;

    Dscalar2 vlast,vnext;

    int cellIdx = d_vertexCellNeighbors[fsidx];
    Dscalar Adiff = KA*(d_AreaPerimeter[cellIdx].x - d_AreaPerimeterPreferences[cellIdx].x);
    Dscalar Pdiff = KP*(d_AreaPerimeter[cellIdx].y - d_AreaPerimeterPreferences[cellIdx].y);

    //vcur = d_voroCur[fsidx];
    vlast.x = d_voroLastNext[fsidx].x;
    vlast.y = d_voroLastNext[fsidx].y;
    vnext.x = d_voroLastNext[fsidx].z;
    vnext.y = d_voroLastNext[fsidx].w;
    computeForceSetAVM(d_voroCur[fsidx],vlast,vnext,Adiff,Pdiff,d_vertexForceSets[fsidx]);
    };

/*!
  the force on a vertex is decomposable into the force contribution from each of its voronoi
  vertices... add 'em up!
  */
__global__ void avm_sum_force_sets_kernel(
                                    Dscalar2*  d_vertexForceSets,
                                    Dscalar2*  d_vertexForces,
                                    int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    Dscalar2 ftemp;
    ftemp.x = 0.0; ftemp.y=0.0;
    for (int ff = 0; ff < 3; ++ff)
        {
        ftemp.x += d_vertexForceSets[3*idx+ff].x;
        ftemp.y += d_vertexForceSets[3*idx+ff].y;
        };
    d_vertexForces[idx] = ftemp;
    };

/*!
  In this version of the active vertex model, the motility of a vertex is a straight average of the
  motility of the three adjacent cells
  */
__global__ void avm_displace_vertices_kernel(
                                        Dscalar2 *d_vertexPositions,
                                        Dscalar2 *d_vertexForces,
                                        Dscalar  *d_cellDirectors,
                                        int      *d_vertexCellNeighbors,
                                        Dscalar  v0,
                                        Dscalar  deltaT,
                                        gpubox   Box,
                                        int      Nvertices)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nvertices)
        return;

    //the vertex motility is the average of th motility of the connected cells
    int vn1 = d_vertexCellNeighbors[3*idx];
    int vn2 = d_vertexCellNeighbors[3*idx+1];
    int vn3 = d_vertexCellNeighbors[3*idx+2];
    Dscalar directorx =
            (Cos(d_cellDirectors[vn1])+Cos(d_cellDirectors[vn2])+Cos(d_cellDirectors[vn3]))/3.0;
    Dscalar directory =
            (Sin(d_cellDirectors[vn1])+Sin(d_cellDirectors[vn2])+Sin(d_cellDirectors[vn3]))/3.0;
    //update positions from forces and motility


//    printf("cell %f\t %f\n",deltaT*(v0*directorx), deltaT*d_vertexForces[idx].x);


    d_vertexPositions[idx].x += deltaT*(v0*directorx + d_vertexForces[idx].x);
    d_vertexPositions[idx].y += deltaT*(v0*directory + d_vertexForces[idx].y);
    //make sure the vertices stay in the box
    Box.putInBoxReal(d_vertexPositions[idx]);
    };

/*!
  After the vertices have been moved, the directors of the cells have some noise.
  */
__global__ void avm_rotate_directors_kernel(
                                        Dscalar  *d_cellDirectors,
                                        curandState *d_curandRNGs,
                                        Dscalar  Dr,
                                        Dscalar  deltaT,
                                        int      Ncells)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;

    //get the per-cell RNG, rotate the director, return the RNG
    curandState_t randState;
    randState=d_curandRNGs[idx];
    d_cellDirectors[idx] += cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    d_curandRNGs[idx] = randState;
    };

/*!
  There will be severe topology mismatches if a cell is involved in more than one T1 transition
  simultaneously (due to incoherent updates of the cellVertices structure). So, go through the
  current list of edges that are marked to take part in a T1 transition and select one edge per
  cell to be flipped on this trip through the functions.
  */
__global__ void avm_one_T1_per_cell_per_vertex_kernel(
                                        int* __restrict__ d_vertexEdgeFlips,
                                        int* __restrict__ d_vertexEdgeFlipsCurrent,
                                        const int* __restrict__ d_vertexNeighbors,
                                        const int* __restrict__ d_vertexCellNeighbors,
                                        const int* __restrict__ d_cellVertexNum,
                                        const int * __restrict__ d_cellVertices,
                                        int *d_finishedFlippingEdges,
                                        Index2D n_idx,
                                        int Ncells)
    {
    unsigned int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= Ncells)
        return;

    //look through every vertex of the cell
    int cneigh = d_cellVertexNum[cell];
    int vertex;
    bool skipRestOfCell = false;
    for (int cc = 0; cc < cneigh; ++cc)
        {
        if (skipRestOfCell) continue;
        vertex = d_cellVertices[n_idx(cc,cell)];
        //what are the other cells attached to this vertex? For correctness, only one cell should
        //own each vertex here. For simplicity, only the lowest-indexed cell gets to do any work.
        if(d_vertexCellNeighbors[3*vertex] < cell ||
               d_vertexCellNeighbors[3*vertex+1] < cell ||
               d_vertexCellNeighbors[3*vertex+2] < cell)
            continue;

        if(d_vertexEdgeFlips[3*vertex] == 1)
            {
            d_vertexEdgeFlipsCurrent[3*vertex] = 1;
            d_vertexEdgeFlips[3*vertex] = 0;
            skipRestOfCell = true;
            };
        if (skipRestOfCell) continue;
        if(d_vertexEdgeFlips[3*vertex+1] == 1)
            {
            d_vertexEdgeFlipsCurrent[3*vertex+1] = 1;
            d_vertexEdgeFlips[3*vertex+1] = 0;
            skipRestOfCell = true;
            };
        if (skipRestOfCell) continue;
        if(d_vertexEdgeFlips[3*vertex+2] == 1)
            {
            d_vertexEdgeFlipsCurrent[3*vertex+2] = 1;
            d_vertexEdgeFlips[3*vertex+2] = 0;
            skipRestOfCell = true;
            };
        };
    if (skipRestOfCell)
        d_finishedFlippingEdges[0] = 1;

    };


/*!
  Run through every pair of vertices (once), see if any T1 transitions should be done,
  and see if the cell-vertex list needs to grow
  */
__global__ void avm_simple_T1_test_kernel(Dscalar2* d_vertexPositions,
                                        int      *d_vertexNeighbors,
                                        int      *d_vertexEdgeFlips,
                                        int      *d_vertexCellNeighbors,
                                        int      *d_cellVertexNum,
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
    int vertex2 = d_vertexNeighbors[idx];
    Dscalar2 edge;
    if(vertex1 < vertex2)
        {
        Box.minDist(d_vertexPositions[vertex1],d_vertexPositions[vertex2],edge);
        if(norm(edge) < T1THRESHOLD)
            {
            d_vertexEdgeFlips[idx]=1;


            //test the number of neighbors of the cells connected to v1 and v2 to see if the
            //cell list should grow this is kind of slow, and I wish I could optimize it away,
            //or at least not test for it during every time step. The latter seems pretty doable.
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1+2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2+2]] == vertexMax)
                d_grow[0] = 1;
            }
        else
            d_vertexEdgeFlips[idx]=0;
        }
    else
        d_vertexEdgeFlips[idx] = 0;

    };

/*!
  Flip any edge labeled for re-wiring in the vertexEdgeFlipsCurrent list
  */
__global__ void avm_flip_edges_kernel(int* d_vertexEdgeFlipsCurrent,
                                      Dscalar2 *d_vertexPositions,
                                      int      *d_vertexNeighbors,
                                      int      *d_vertexCellNeighbors,
                                      int      *d_cellVertexNum,
                                      int      *d_cellVertices,
                                      int      *d_finishedFlippingEdges,
                                      Dscalar  T1Threshold,
                                      gpubox   Box,
                                      Index2D  n_idx,
                                      int      NvTimes3)
    {
    if (d_finishedFlippingEdges[0]==0) return;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //return if the index is out of bounds or if the edge isn't marked for flipping
    if (idx >= NvTimes3 || d_vertexEdgeFlipsCurrent[idx] == 0)
        return;
    //identify the vertices and reset the flag
    int vertex1 = idx/3;
    int vertex2 = d_vertexNeighbors[idx];
    d_vertexEdgeFlipsCurrent[idx] = 0;

//printf("T1 for vertices %i %i ...\n",vertex1,vertex2);

    //Rotate the vertices in the edge and set them at twice their original distance
    Dscalar2 edge;
    Dscalar2 v1 = d_vertexPositions[vertex1];
    Dscalar2 v2 = d_vertexPositions[vertex2];
    Box.minDist(v1,v2,edge);
    if(norm(edge) < T1Threshold) return;

    //Dscalar2 midpoint;
    //midpoint.x = v2.x + 0.5*edge.x;
    //midpoint.y = v2.y + 0.5*edge.y;

    //v1.x = midpoint.x-edge.y;v1.y = midpoint.y+edge.x;
    //v2.x = midpoint.x+edge.y;v2.y = midpoint.y-edge.x;
    v1.x = v2.x + 0.5*edge.x-edge.y;
    v1.y = v2.y + 0.5*edge.y+edge.x;
    v2.x = v2.x + 0.5*edge.x+edge.y;
    v2.y = v2.y + 0.5*edge.y-edge.x;
    Box.putInBoxReal(v1);
    Box.putInBoxReal(v2);
    d_vertexPositions[vertex1] = v1;
    d_vertexPositions[vertex2] = v2;

    //now, do the gross work of cell and vertex rewiring
    int4 cellSet;cellSet.x=-1;cellSet.y=-1;cellSet.z=-1;cellSet.w=-1;
    //int4 vertexSet;
    int2 vertexSet;
    ///////////////////////////////////////////////////
    //TERRIBLE GPU CODE = COPY THE CPU BRANCH LOGIC....
    ///////////////////////////////////////////////////
    int cell1,cell2,cell3,ctest;
    int vlast, vcur, vnext, cneigh;
    cell1 = d_vertexCellNeighbors[3*vertex1];
    cell2 = d_vertexCellNeighbors[3*vertex1+1];
    cell3 = d_vertexCellNeighbors[3*vertex1+2];
    //cell_l doesn't contain vertex 1, so it is the cell neighbor of vertex 2 we haven't found yet
    for (int ff = 0; ff < 3; ++ff)
        {
        ctest = d_vertexCellNeighbors[3*vertex2+ff];
        if(ctest != cell1 && ctest != cell2 && ctest != cell3)
            cellSet.w=ctest;
        };
    //find vertices "c" and "d"
    cneigh = d_cellVertexNum[cellSet.w];
    vlast = d_cellVertices[ n_idx(cneigh-2,cellSet.w) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell1)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };

    //classify cell1
    cneigh = d_cellVertexNum[cell1];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell1) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell1) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell1)];
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
    cneigh = d_cellVertexNum[cell2];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell2) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell2) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell2)];
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
    cneigh = d_cellVertexNum[cell3];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell3) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell3) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell3)];
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
    cneigh = d_cellVertexNum[cellSet.y];
    vlast = d_cellVertices[ n_idx(cneigh-2,cellSet.y) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cellSet.y) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cellSet.y)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    //vertexSet.x=vlast;
    //vertexSet.y=vnext;
    vertexSet.x=vnext;
    cneigh = d_cellVertexNum[cellSet.w];
    vlast = d_cellVertices[ n_idx(cneigh-2,cellSet.w) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cellSet.w)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };
    //vertexSet.w=vlast;
    //vertexSet.z=vnext;
    vertexSet.y=vnext;
    ///////////////////////////////////////////////////
    //END OF FIRST CHUNK OF TERRIBLE CODE...but the nightmare isn't over
    ///////////////////////////////////////////////////

    //re-wire the cells and vertices
    //start with the vertex-vertex and vertex-cell  neighbors
    for (int vert = 0; vert < 3; ++vert)
        {
        //vertex-cell neighbors
        if(d_vertexCellNeighbors[3*vertex1+vert] == cellSet.z)
            d_vertexCellNeighbors[3*vertex1+vert] = cellSet.w;
        if(d_vertexCellNeighbors[3*vertex2+vert] == cellSet.x)
            d_vertexCellNeighbors[3*vertex2+vert] = cellSet.y;
        //vertex-vertex neighbors
        if(d_vertexNeighbors[3*vertexSet.x+vert] == vertex1)
            d_vertexNeighbors[3*vertexSet.x+vert] = vertex2;
        if(d_vertexNeighbors[3*vertexSet.y+vert] == vertex2)
            d_vertexNeighbors[3*vertexSet.y+vert] = vertex1;
        if(d_vertexNeighbors[3*vertex1+vert] == vertexSet.x)
            d_vertexNeighbors[3*vertex1+vert] = vertexSet.y;
        if(d_vertexNeighbors[3*vertex2+vert] == vertexSet.y)
            d_vertexNeighbors[3*vertex2+vert] = vertexSet.x;
        };
    //now rewire the cells
    //cell i loses v2 as a neighbor

    cneigh = d_cellVertexNum[cellSet.x];
    int cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cellVertices[n_idx(cc,cellSet.x)] == vertex2)
            cidx +=1;
        d_cellVertices[n_idx(cc,cellSet.x)] = d_cellVertices[n_idx(cidx,cellSet.x)];
        cidx +=1;
        };
    d_cellVertexNum[cellSet.x] -= 1;

    //cell j gains v2 in between v1 and b, so step through list backwards and insert
    cneigh = d_cellVertexNum[cellSet.y];
    cidx = cneigh;
    int vLocation = cneigh;
    for (int cc = cneigh-1;cc >=0; --cc)
        {
        int cellIndex = d_cellVertices[n_idx(cc,cellSet.y)];
        if(cellIndex == vertex1)
            {
            vLocation = cidx;
            cidx -= 1;
            };
        d_cellVertices[n_idx(cidx,cellSet.y)] = cellIndex;
        cidx -= 1;
        };
    if(cidx ==0)
        d_cellVertices[n_idx(0,cellSet.y)] = vertex2;
    else
        d_cellVertices[n_idx(vLocation,cellSet.y)] = vertex2;
    d_cellVertexNum[cellSet.y] += 1;

    //cell k loses v1 as a neighbor
    cneigh = d_cellVertexNum[cellSet.z];
    cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cellVertices[n_idx(cc,cellSet.z)] == vertex1)
            cidx +=1;
        d_cellVertices[n_idx(cc,cellSet.z)] = d_cellVertices[n_idx(cidx,cellSet.z)];
        cidx +=1;
        };
    d_cellVertexNum[cellSet.z] -= 1;

    //cell l gains v1 in between v2 and c...copy the logic of cell j
    cneigh = d_cellVertexNum[cellSet.w];
    cidx = cneigh;
    vLocation = cneigh;
    for (int cc = cneigh-1;cc >=0; --cc)
        {
        int cellIndex = d_cellVertices[n_idx(cc,cellSet.w)];
        if(cellIndex == vertex2)
            {
            vLocation = cidx;
            cidx -= 1;
            };
        d_cellVertices[n_idx(cidx,cellSet.w)] = cellIndex;
        cidx -= 1;
        };
    if(cidx ==0)
        d_cellVertices[n_idx(0,cellSet.w)] = vertex1;
    else
        d_cellVertices[n_idx(vLocation,cellSet.w)] = vertex1;
    d_cellVertexNum[cellSet.w] += 1;

    ///////////////////////////////////////////////////
    //END OF COPIED CODE
    ///////////////////////////////////////////////////
    };


/*!
  This function is being deprecated, but is still useful for calculating, e.g. the mean-squared
displacement of the cells without transferring data back to the hose
*/
__global__ void avm_get_cell_positions_kernel(Dscalar2* d_cellPositions,
                                              Dscalar2* d_vertexPositions,
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
    baseVertex = d_vertexPositions[ d_n[n_idx(0,idx)] ];
    int neighs = d_nn[idx];
    for (int n = 1; n < neighs; ++n)
        {
        Box.minDist(d_vertexPositions[ d_n[n_idx(n,idx)] ],baseVertex,vertex);
        pos.x += vertex.x;
        pos.y += vertex.y;
        };
    pos.x /= neighs;
    pos.y /= neighs;
    pos.x += baseVertex.x;
    pos.y += baseVertex.y;
    Box.putInBoxReal(pos);
    d_cellPositions[idx] = pos;
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
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_vertexCellNeighbors,
                    Dscalar2 *d_voroCur,
                    Dscalar4 *d_voroLastNext,
                    Dscalar2 *d_AreaPerimeter,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    avm_geometry_kernel<<<nblocks,block_size>>>(d_vertexPositions,
                                                d_cellVertexNum,d_cellVertices,
                                                d_vertexCellNeighbors,d_voroCur,
                                                d_voroLastNext,d_AreaPerimeter,
                                                N, n_idx, Box);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to calculate force sets
bool gpu_avm_force_sets(
                    int      *d_vertexCellNeighbors,
                    Dscalar2 *d_voroCur,
                    Dscalar4 *d_voroLastNext,
                    Dscalar2 *d_AreaPerimeter,
                    Dscalar2 *d_AreaPerimeterPreferences,
                    Dscalar2 *d_vertexForceSets,
                    int nForceSets,
                    Dscalar KA, Dscalar KP)
    {
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks  = nForceSets/block_size + 1;

    avm_force_sets_kernel<<<nblocks,block_size>>>(d_vertexCellNeighbors,d_voroCur,d_voroLastNext,
                                                  d_AreaPerimeter,d_AreaPerimeterPreferences,
                                                  d_vertexForceSets,
                                                  nForceSets,KA,KP);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to sum up the force sets to get net force on each vertex
bool gpu_avm_sum_force_sets(
                    Dscalar2 *d_vertexForceSets,
                    Dscalar2 *d_vertexForces,
                    int      Nvertices)
    {
    unsigned int block_size = 128;
    if (Nvertices < 128) block_size = 32;
    unsigned int nblocks  = Nvertices/block_size + 1;


    avm_sum_force_sets_kernel<<<nblocks,block_size>>>(d_vertexForceSets,d_vertexForces,Nvertices);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_avm_displace_and_rotate(
                    Dscalar2 *d_vertexPositions,
                    Dscalar2 *d_vertexForces,
                    Dscalar  *d_cellDirectors,
                    int      *d_vertexCellNeighbors,
                    curandState *d_curandRNGs,
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
    avm_displace_vertices_kernel<<<nblocks,block_size>>>(d_vertexPositions,d_vertexForces,
                                                         d_cellDirectors,d_vertexCellNeighbors,
                                                         v0,deltaT,Box,Nvertices);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    //rotate cell directors
    if (Ncells < 128) block_size = 32;
    nblocks = Ncells/block_size + 1;
    avm_rotate_directors_kernel<<<nblocks,block_size>>>(d_cellDirectors,d_curandRNGs,
                                                        Dr,deltaT,Ncells);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    return cudaSuccess;
    };


//!Call the kernel to test every edge for a T1 event, see if vertexMax needs to increase
bool gpu_avm_test_edges_for_T1(
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    gpubox   &Box,
                    Dscalar  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow,
                    Index2D  &n_idx)
    {
    unsigned int block_size = 128;
    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    unsigned int nblocks  = NvTimes3/block_size + 1;

    //test edges
    avm_simple_T1_test_kernel<<<nblocks,block_size>>>(
                                                      d_vertexPositions,d_vertexNeighbors,
                                                      d_vertexEdgeFlips,d_vertexCellNeighbors,
                                                      d_cellVertexNum,
                                                      Box,T1THRESHOLD,
                                                      NvTimes3,vertexMax,d_grow);

    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to flip at most one edge per cell, write to d_finishedFlippingEdges the current state
bool gpu_avm_flip_edges(
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexEdgeFlipsCurrent,
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_finishedFlippingEdges,
                    Dscalar  T1Threshold,
                    gpubox   &Box,
                    Index2D  &n_idx,
                    int      Nvertices,
                    int      Ncells)
    {
    unsigned int block_size = 128;

    /*The issue is that if a cell is involved in two edge flips done by different threads, the resulting
    data structure for what vertices belong to cells and what cells border which vertex will be
    inconsistently updated.

    The strategy will be to take the d_vertexEdgeFlips list, put at most one T1 per cell per vertex into the
    d_vertexEdgeFlipsCurrent list (erasing it from the d_vertexEdgeFlips list), and swap the edges specified
    by the "current" list. If d_vertexEdgeFlips is empty, we will set d_finishedFlippingEdges to 1. As long
    as it is != 1, the cpp code will continue calling this gpu_avm_flip_edges function.
    */

    //first select a few edges to flip...
    if(Ncells <128) block_size = 32;
    unsigned int nblocks = Ncells/block_size + 1;
    avm_one_T1_per_cell_per_vertex_kernel<<<nblocks,block_size>>>(
                                                                d_vertexEdgeFlips,
                                                                d_vertexEdgeFlipsCurrent,
                                                                d_vertexNeighbors,
                                                                d_vertexCellNeighbors,
                                                                d_cellVertexNum,
                                                                d_cellVertices,
                                                                d_finishedFlippingEdges,
                                                                n_idx,
                                                                Ncells);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    //Now flip 'em
    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    nblocks  = NvTimes3/block_size + 1;

    avm_flip_edges_kernel<<<nblocks,block_size>>>(
                                                  d_vertexEdgeFlipsCurrent,d_vertexPositions,d_vertexNeighbors,
                                                  d_vertexCellNeighbors,d_cellVertexNum,d_cellVertices,
                                                  d_finishedFlippingEdges,
                                                  T1Threshold,Box,
                                                  n_idx,NvTimes3);

    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


//!Call the kernel to calculate the position of each cell from the position of its vertices
bool gpu_avm_get_cell_positions(
                    Dscalar2 *d_cellPositions,
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    avm_get_cell_positions_kernel<<<nblocks,block_size>>>(d_cellPositions,d_vertexPositions,
                                                          d_cellVertexNum,d_cellVertices,
                                                          N, n_idx, Box);
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
