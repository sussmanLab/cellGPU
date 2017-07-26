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
    curandState_t randState;

    randState=RNGs[idx];
    Dscalar dirx = Cos(cellDirectors[idx]);
    Dscalar diry = Sin(cellDirectors[idx]);
    Dscalar v0 = motility[idx].x;
    Dscalar Dr = motility[idx].y;
    Dscalar angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    cellDirectors[idx] += angleDiff;

    RNGs[idx] = randState;

    displacements[idx].x = deltaT*(v0*dirx + mu*forces[idx].x);
    displacements[idx].y = deltaT*(v0*diry + mu*forces[idx].y);

    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_spp_eom_integration(
                    Dscalar2 *forces,
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
                                forces,displacements,motility,cellDirectors,
                                RNGs,
                                N,deltaT,Timestep,mu);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration

