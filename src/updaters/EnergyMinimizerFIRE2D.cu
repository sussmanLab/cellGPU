#define NVCC
#define ENABLE_CUDA

#include "EnergyMinimizerFIRE2D.cuh"

/*! \file EnergyMinimizerFIRE2D.cu
  defines kernel callers and kernels for GPU calculations related to FIRE minimization

 \addtogroup EnergyMinimizerFIRE2DKernels
 @{
 */

/*!
  set the first N elements of the d_velocity vector to 0.0
*/
__global__ void gpu_zero_velocity_kernel(Dscalar2 *d_velocity,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    d_velocity[idx].x = 0.0;
    d_velocity[idx].y = 0.0;
    return;
    };


/*!
update the velocity in a velocity Verlet step
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

/*!
update the velocity according to a FIRE step
*/
__global__ void gpu_update_velocity_FIRE_kernel(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar alpha, Dscalar scaling, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx].x = (1-alpha)*d_velocity[idx].x + alpha*scaling*d_force[idx].x;
    d_velocity[idx].y = (1-alpha)*d_velocity[idx].y + alpha*scaling*d_force[idx].y;
    };

/*!
calculate the displacement in a velocity verlet step according to the force and velocity
*/
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
  \param d_velocity the GPU array data of the velocities
  \param N length of the array
  \post all elements of d_velocity are set to (0.0,0.0)
  */
bool gpu_zero_velocity(Dscalar2 *d_velocity,
                    int N
                    )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_zero_velocity_kernel<<<nblocks, block_size>>>(d_velocity,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/*!
\param d_velocity Dscalar2 array of velocity
\param d_force Dscalar2 array of force
\param alpha the FIRE parameter
\param scaling the square root of (v.v / f.f)
\param N      the length of the arrays
\post v = (1-alpha)v + alpha*scalaing*force
*/
bool gpu_update_velocity_FIRE(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar alpha, Dscalar scaling, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_FIRE_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                alpha,
                                                scaling,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_velocity Dscalar2 array of velocity
\param d_force Dscalar2 array of force
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
