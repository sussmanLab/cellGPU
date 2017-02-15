#define NVCC
#define ENABLE_CUDA

#include "EnergyMinimizerFIRE2D.cuh"

/*! \file EnergyMinimizerFIRE2D.cu
  defines kernel callers and kernels for GPU calculations related to FIRE minimization

 \addtogroup EnergyMinimizerFIRE2DKernels
 @{
 */

__global__ void gpu_update_velocity_kernel(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx].x += 0.5*deltaT*d_force[idx].x;
    d_velocity[idx].y += 0.5*deltaT*d_force[idx].y;
    };

__global__ void gpu_displacement_vv_kernel(Dscalar2 *d_displacement, Dscalar2 *d_velocity,
                                           Dscalar2 *d_force, Dscalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_displacement[idx].x = deltaT*d_velocity[idx].x+0.5*deltaT*deltaT*d_force[idx].x;
    d_displacement[idx].y = deltaT*d_velocity[idx].y+0.5*deltaT*deltaT*d_force[idx].y;
    };


/*!
\param d_velocity Dscalar array of velocity
\param d_force Dscalar array of force
\param deltaT time step
\param N      the length of the arrays
\post v = v + 0.5*deltaT*force
*/
bool gpu_update_velocity(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar deltaT, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                deltaT,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_displacement Dscalar2 array of displacements
\param d_velocity Dscalar2 array of velocities
\param d_force Dscalar2 array of forces
\param Dscalar deltaT the current time step
\param N      the length of the arrays
\post displacement = dt*velocity + 0.5 *dt^2*force
*/
bool gpu_displacement_velocity_verlet(Dscalar2 *d_displacement,
                      Dscalar2 *d_velocity,
                      Dscalar2 *d_force,
                      Dscalar deltaT,
                      int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_displacement_vv_kernel<<<nblocks,block_size>>>(
                                                d_displacement,d_velocity,d_force,deltaT,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


/** @} */ //end of group declaration
