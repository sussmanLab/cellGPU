#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "vertexQuadraticEnergy.cuh"

/** \file vertexQuadraticEnergy.cu
    * Defines kernel callers and kernels for GPU calculations of vertex model parts
*/

/*!
    \addtogroup vmKernels
    @{
*/

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
    computeForceSetVertexModel(d_voroCur[fsidx],vlast,vnext,Adiff,Pdiff,d_vertexForceSets[fsidx]);
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
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
