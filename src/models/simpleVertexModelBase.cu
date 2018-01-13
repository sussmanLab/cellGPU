#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "vertexModelBase.cuh"

/** \file simpleVertexModelBase.cu
    * Defines kernel callers and kernels for GPU calculations of vertex models
*/

/*!
    \addtogroup vmKernels
    @{
*/

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
