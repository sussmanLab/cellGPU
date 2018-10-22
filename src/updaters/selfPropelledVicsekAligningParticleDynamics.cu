#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "selfPropelledVicsekAligningParticleDynamics.cuh"

/** \file selfPropelledVicsekAligningParticleDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

/*!
Each thread calculates the displacement of an individual cell
*/
__global__ void spp_vicsek_aligning_eom_integration_kernel(Dscalar2 *forces,
                                           Dscalar2 *velocities,
                                           Dscalar2 *displacements,
                                           Dscalar2 *motility,
                                           Dscalar *cellDirectors,
                                           int *nNeighbors,
                                           int *neighbors,
                                           Index2D  n_idx,
                                           curandState *RNGs,
                                           int N,
                                           Dscalar deltaT,
                                           int Timestep,
                                           Dscalar mu,
                                           Dscalar Eta)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    //get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    Dscalar v0 = motility[idx].x;
    //Dscalar Dr = motility[idx].y;
    Dscalar randomAngle = 2.0*PI*curand_uniform(&randState);
    RNGs[idx] = randState;

    Dscalar currentTheta = cellDirectors[idx];
    //update displacements
    velocities[idx].x = v0*Cos(currentTheta) + mu*forces[idx].x;
    velocities[idx].y = v0*Sin(currentTheta) + mu*forces[idx].y;
    displacements[idx] = deltaT*velocities[idx];

    Dscalar2 direction; direction.x = 0.0; direction.y=0.0;
    int neigh = nNeighbors[idx];
    for (int nn =0; nn < neigh; ++nn)
        {
        Dscalar curTheta = cellDirectors[neighbors[n_idx(nn,idx)]];
        direction.x += Cos(curTheta);
        direction.y += Sin(curTheta);
        }
    direction.x += neigh*Eta*Cos(randomAngle);
    direction.y += neigh*Eta*Sin(randomAngle);
    Dscalar phi = atan2(direction.y,direction.x);
    
    //update director
    cellDirectors[idx] = phi;

    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_spp_vicsek_aligning_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *velocities,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar *cellDirectors,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu,
                    Dscalar Eta)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    spp_vicsek_aligning_eom_integration_kernel<<<nblocks,block_size>>>(
                                forces,velocities,displacements,motility,cellDirectors,
                                nNeighbors,neighbors,n_idx,
                                RNGs,
                                N,deltaT,Timestep,mu, Eta);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
