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
  A simple routine that takes in a pointer array of points, an array of displacements,
  adds the displacements to the points, but with the displacement vector scaled by some amount, and
  puts the points back in the primary unit cell.
  This is useful, e.g., when the displacements are a dt times a velocity
*/
__global__ void gpu_move_degrees_of_freedom_kernel(Dscalar2 *d_points,
                                          Dscalar2 *d_disp,
                                          Dscalar scale,
                                          int N,
                                          gpubox Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_points[idx].x += scale*d_disp[idx].x;
    d_points[idx].y += scale*d_disp[idx].y;
    Box.putInBoxReal(d_points[idx]);
    return;
    };

/*!
every thread just writes in a value
*/
__global__ void gpu_set_integer_array_kernel(int *d_array,
                                          int value,
                                          int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_array[idx] = value;
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
                        Dscalar  scale,
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
                                                scale,
                                                N,
                                                Box
                                                );
    HANDLE_ERROR(cudaGetLastError());

    return cudaSuccess;
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
    HANDLE_ERROR(cudaGetLastError());

    return cudaSuccess;
    };

/*!
\param d_array int array of values
\param value   the integer to set the entire array to
\param N        The number of values in the array to set (d_array[0] tp d_array[N-1])
*/
bool gpu_set_integer_array(int *d_array,
                           int value,
                           int N
                          )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_set_integer_array_kernel<<<nblocks,block_size>>>(
                                                d_array,
                                                value,
                                                N);
    HANDLE_ERROR(cudaGetLastError());

    return cudaSuccess;
    };

/** @} */ //end of group declaration
