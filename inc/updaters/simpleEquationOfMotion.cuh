#ifndef __SIMPLEEQUATIONOFMOTION_CUH__
#define __SIMPLEEQUATIONOFMOTION_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file simpleEquationOfMotion.cuh
A file providing an interface to the relevant cuda calls for the simpleEquationOfMotion class
*/

/** @addtogroup selfPropelledParticleDynamicsKernels selfPropelledParticleDynamics Kernels
 * @{
 * \brief CUDA kernels and callers for the simpleEquationsOfMotion and childe classes
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_RNG(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );
/** @} */ //end of group declaration
#endif
