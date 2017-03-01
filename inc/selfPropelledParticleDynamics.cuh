#ifndef __SELFPROPELLEDPARTICLEDYNAMICS_CUH__
#define __SELFPROPELLEDPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the selfPropelledParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_sppRNG(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

//!set the vector of displacements from forces and activity
bool gpu_spp_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar *cellDirectors,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu);

/** @} */ //end of group declaration
 #endif

