#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "selfPropelledAligningParticleDynamics.cuh"

/** \file selfPropelledAligningParticleDynamics.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/

/*!
Each thread calculates the displacement of an individual cell
*/
__global__ void spp_aligning_eom_integration_kernel(double2 *forces,
                                           double2 *velocities,
                                           double2 *displacements,
                                           double2 *motility,
                                           double *cellDirectors,
                                           curandState *RNGs,
                                           int N,
                                           double deltaT,
                                           int Timestep,
                                           double mu,
                                           double J)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    //get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    double v0 = motility[idx].x;
    double Dr = motility[idx].y;
    double angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    RNGs[idx] = randState;

    double currentTheta = cellDirectors[idx];
    if(currentTheta < -PI)
        currentTheta += 2*PI;
    if(currentTheta > PI)
        currentTheta -= 2*PI;
    //update displacements
    velocities[idx].x = v0*Cos(currentTheta) + mu*forces[idx].x;
    velocities[idx].y = v0*Sin(currentTheta) + mu*forces[idx].y;
    displacements[idx] = deltaT*velocities[idx];

    double currentPhi = atan2(displacements[idx].y,displacements[idx].x);

    //update director
    cellDirectors[idx] = currentTheta + angleDiff - deltaT*J*Sin(currentTheta-currentPhi);
    return;
    };

//!get the current timesteps vector of displacements into the displacement vector
bool gpu_spp_aligning_eom_integration(
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double2 *motility,
                    double *cellDirectors,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    int Timestep,
                    double mu,
                    double J)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    spp_aligning_eom_integration_kernel<<<nblocks,block_size>>>(
                                forces,velocities,displacements,motility,cellDirectors,
                                RNGs,
                                N,deltaT,Timestep,mu, J);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
