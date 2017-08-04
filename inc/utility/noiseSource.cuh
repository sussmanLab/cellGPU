#ifndef __NOISESOURCE_CUH__
#define __NOISESOURCE_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file noiseSource.cuh
A file providing an interface to the relevant cuda calls for the simpleEquationOfMotion class
*/

/** @addtogroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for generating rngs on the gpu
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_RNG_array(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );
/** @} */ //end of group declaration
#endif
