#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "langevinDynamics.cuh"

/** \file langevinDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple brownian 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

/*!
Each thread calculates the displacement of an individual cell
*/
__global__ void langevin_BandO_kernel(
                                    double2 *velocities,
                                    double2 *forces,
                                    double2 *displacements,
                                    curandState *RNGs,
                                    int N,
                                    double deltaT,
                                    double gamma,
                                    double Temperature)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    curandState_t randState;
    randState=RNGs[idx];

    velocities[idx] = velocities[idx] + (0.5*deltaT)*forces[idx];
    displacements[idx] = (0.5*deltaT)*velocities[idx];
    double c1 = exp(-gamma*deltaT);
    double c2 = sqrt(Temperature)*sqrt(1.0-c1*c1);
    velocities[idx].x = c1*velocities[idx].x + cur_norm(&randState)*c2;
    velocities[idx].y = c1*velocities[idx].y + cur_norm(&randState)*c2;

    RNGs[idx] = randState;
    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_langevin_BandO_operation(
                    double2 *velocities,
                    double2 *forces,
                    double2 *displacements,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    double gamma,
                    double T)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    langevin_BandO_kernel<<<nblocks,block_size>>>(
                                velocities,forces,displacements,
                                RNGs,
                                N,deltaT,gamma,T);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
