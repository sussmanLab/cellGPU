#ifndef __SELFPROPELLEDPARTICLEDYNAMICS_CUH__
#define __SELFPROPELLEDPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the Simple2DActiveCell class
*/

/** @defgroup selfPropelledParticleDynamicsKernels selfPropelledParticleDynamics Kernels
 * @{
 * \brief CUDA kernels and callers for the selfPropelledParticleDynamics class
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_sppRNG(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

/** @} */ //end of group declaration
 #endif

