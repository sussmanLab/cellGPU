#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "vertexModelBase.cuh"

/** \file vertexModelBase.cu
    * Defines kernel callers and kernels for GPU calculations of vertex models
*/

/*!
    \addtogroup vmKernels
    @{
*/

/*!
  Since the cells are NOT guaranteed to be convex, the area of the cell must take into account any
  self-intersections. The strategy is the same as in the CPU branch.
  */
__global__ void vm_geometry_kernel(
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
  This function is being deprecated, but is still useful for calculating, e.g. the mean-squared
displacement of the cells without transferring data back to the host
*/
__global__ void vm_get_cell_positions_kernel(Dscalar2* d_cellPositions,
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

/*!
  This kernel just moves the vertices around according to the input vector, then makes sure the
  vertices stay in the box.
  */
__global__ void vm_move_vertices_kernel(
                                        Dscalar2 *d_vertexPositions,
                                        Dscalar2 *d_vertexDisplacements,
                                        gpubox   Box,
                                        int      Nvertices)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nvertices)
        return;

    d_vertexPositions[idx].x += d_vertexDisplacements[idx].x;
    d_vertexPositions[idx].y += d_vertexDisplacements[idx].y;
    //make sure the vertices stay in the box
    Box.putInBoxReal(d_vertexPositions[idx]);
    };

//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_vm_geometry(
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


    vm_geometry_kernel<<<nblocks,block_size>>>(d_vertexPositions,
                                               d_cellVertexNum,d_cellVertices,
                                               d_vertexCellNeighbors,d_voroCur,
                                               d_voroLastNext,d_AreaPerimeter,
                                               N, n_idx, Box);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to displace vertices according to the displacement vector
bool gpu_vm_displace(
                    Dscalar2 *d_vertexPositions,
                    Dscalar2 *d_vertexDisplacements,
                    gpubox   &Box,
                    int      Nvertices)
    {
    unsigned int block_size = 128;
    if (Nvertices < 128) block_size = 32;
    unsigned int nblocks  = Nvertices/block_size + 1;

    vm_move_vertices_kernel<<<nblocks,block_size>>>(d_vertexPositions,d_vertexDisplacements,
                                                     Box,Nvertices);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to calculate the position of each cell from the position of its vertices
bool gpu_vm_get_cell_positions(
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


    vm_get_cell_positions_kernel<<<nblocks,block_size>>>(d_cellPositions,d_vertexPositions,
                                                          d_cellVertexNum,d_cellVertices,
                                                          N, n_idx, Box);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
