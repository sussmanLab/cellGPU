#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "Simple2DCell.cuh"

/** \file Simple2DCell.cu
    * Defines kernel callers and kernels for GPU calculations of simple 2D cell models
*/

/*!
    \addtogroup Simple2DCellKernels
    @{
*/

/*!
  A simple routine that takes in a pointer array of points, an array of displacements,
  adds the displacements to the points, and puts the points back in the primary unit cell.
*/
__global__ void gpu_move_degrees_of_freedom_kernel(Dscalar2 *d_points,
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


/*!
\param d_points Dscalar2 array of locations
\param d_disp   Dscalar2 array of displacements
\param N        The number of degrees of freedom to move
\param Box      The gpubox in which the new positions must reside
*/
bool gpu_move_degrees_of_freedom(Dscalar2 *d_points,
                        Dscalar2 *d_disp,
                        int N,
                        gpubox &Box
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_move_degrees_of_freedom_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_disp,
                                                N,
                                                Box
                                                );
    //cudaThreadSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    return cudaSuccess;
    };




/** @} */ //end of group declaration

