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

__host__ __device__ inline void moveDegreesOfFreedomFunction(int idx, double2 *d_points, double2 *d_disp, periodicBoundaries Box)
    {
    d_points[idx].x += d_disp[idx].x;
    d_points[idx].y += d_disp[idx].y;
    Box.putInBoxReal(d_points[idx]);
    return;
    };
__host__ __device__  inline void moveDegreesOfFreedomFunctionScaled(int idx, double2 *d_points, double2 *d_disp, double scale, periodicBoundaries Box)
    {
    d_points[idx].x += scale*d_disp[idx].x;
    d_points[idx].y += scale*d_disp[idx].y;
    Box.putInBoxReal(d_points[idx]);
    return;
    };


/*!
  A simple routine that takes in a pointer array of points, an array of displacements,
  adds the displacements to the points, and puts the points back in the primary unit cell.
*/
__global__ void gpu_move_degrees_of_freedom_kernel(double2 *d_points,
                                          double2 *d_disp,
                                          int N,
                                          periodicBoundaries Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    moveDegreesOfFreedomFunction(idx,d_points,d_disp,Box);
    return;
    };

/*!
  A simple routine that takes in a pointer array of points, an array of displacements,
  adds the displacements to the points, but with the displacement vector scaled by some amount, and
  puts the points back in the primary unit cell.
  This is useful, e.g., when the displacements are a dt times a velocity
*/
__global__ void gpu_move_degrees_of_freedom_kernel(double2 *d_points,
                                          double2 *d_disp,
                                          double scale,
                                          int N,
                                          periodicBoundaries Box
                                         )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    moveDegreesOfFreedomFunctionScaled(idx,d_points,d_disp,scale,Box);
    return;
    };

/*!
\param d_points double2 array of locations
\param d_disp   double2 array of displacements
\param N        The number of degrees of freedom to move
\param Box      The periodicBoundaries in which the new positions must reside
*/
bool gpu_move_degrees_of_freedom(double2 *d_points,
                        double2 *d_disp,
                        double  scale,
                        int N,
                        periodicBoundaries &Box,
                        bool useGPU,
                        int nThreads
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    
    if(useGPU)
        {
        gpu_move_degrees_of_freedom_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_disp,
                                                scale,
                                                N,
                                                Box
                                                );
        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        }
    else
        {
        ompFunctionLoop(nThreads,N,moveDegreesOfFreedomFunctionScaled,d_points,d_disp,scale,Box);
        }
    return true;
    };

/*!
\param d_points double2 array of locations
\param d_disp   double2 array of displacements
\param N        The number of degrees of freedom to move
\param Box      The periodicBoundaries in which the new positions must reside
*/
bool gpu_move_degrees_of_freedom(double2 *d_points,
                        double2 *d_disp,
                        int N,
                        periodicBoundaries &Box,
                        bool useGPU,
                        int nThreads
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    if(useGPU)
        {
        gpu_move_degrees_of_freedom_kernel<<<nblocks,block_size>>>(
                                                d_points,
                                                d_disp,
                                                N,
                                                Box
                                                );
        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        }
    else
        {
        ompFunctionLoop(nThreads,N,moveDegreesOfFreedomFunction,d_points,d_disp,Box);
        }

    return true;
    };

/** @} */ //end of group declaration
