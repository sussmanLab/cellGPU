#ifndef ENERGYMINIMIZERFIRE2D_CUH__
#define ENERGYMINIMIZERFIRE2D_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "gpubox.h"

/*!
 \file EnergyMinimizerFIRE2D.cuh
A file providing an interface to the relevant cuda calls for the EnergyMinimizerFIRE2D class
*/

/** @defgroup EnergyMinimizerFIRE2DKernels EnergyMinimizerFIRE2D Kernels
 * @{
 * \brief CUDA kernels and callers for the EnergyMinimizerFIRE2D class
 */

//!Zero out the velocity (if the power is negative)
bool gpu_zero_velocity(Dscalar2 *d_velocity,
                       int N);

//!velocity = velocity +0.5*deltaT*force
bool gpu_update_velocity(Dscalar2 *d_velocity,
                      Dscalar2 *d_force,
                      Dscalar deltaT,
                      int N
                      );

//!velocity = (1-a)velocity +a*scaling*force
bool gpu_update_velocity_FIRE(Dscalar2 *d_velocity,
                      Dscalar2 *d_force,
                      Dscalar alpha,
                      Dscalar scaling,
                      int N
                      );


//!displacement = dt*velocity + 0.5*dt^2*force
bool gpu_displacement_velocity_verlet(Dscalar2 *d_displacement,
                      Dscalar2 *d_velocity,
                      Dscalar2 *d_force,
                      Dscalar deltaT,
                      int N
                      );


/** @} */ //end of group declaration

#endif
