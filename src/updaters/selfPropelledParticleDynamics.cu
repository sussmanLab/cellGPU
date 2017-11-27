#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "selfPropelledParticleDynamics.cuh"

/** \file selfPropelledParticleDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

/*!
Each thread calculates the displacement of an individual cell
*/
__global__ void spp_eom_integration_kernel(Dscalar2 *forces,
                                           Dscalar2 *velocities,
                                           Dscalar2 *displacements,
                                           Dscalar2 *motility,
                                           Dscalar *cellDirectors,
                                           curandState *RNGs,
                                           int N,
                                           Dscalar deltaT,
                                           int Timestep,
                                           Dscalar mu)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    //update displacements
    displacements[idx].x = deltaT*(velocities[idx].x + mu*forces[idx].x);
    displacements[idx].y = deltaT*(velocities[idx].y + mu*forces[idx].y);

    //next, get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    Dscalar v0 = motility[idx].x;
    Dscalar Dr = motility[idx].y;
    Dscalar angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    RNGs[idx] = randState;
    //update director and velocity vector
    Dscalar currentTheta = cellDirectors[idx];
    if(velocities[idx].y != 0. && velocities[idx].x != 0.)
        {
        currentTheta = atan2(velocities[idx].y,velocities[idx].x);
        };
    cellDirectors[idx] = currentTheta + angleDiff;
    velocities[idx].x = v0 * Cos(cellDirectors[idx]);
    velocities[idx].y = v0 * Sin(cellDirectors[idx]);


    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_spp_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *velocities,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar *cellDirectors,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    spp_eom_integration_kernel<<<nblocks,block_size>>>(
                                forces,velocities,displacements,motility,cellDirectors,
                                RNGs,
                                N,deltaT,Timestep,mu);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration

