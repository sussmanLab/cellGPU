#ifndef __SIMPLE2DCELL_CUH__
#define __SIMPLE2DCELL_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file Simple3DCell.cuh
A file providing an interface to the relevant cuda calls for the Simple2DCell class
*/

/** @defgroup Simple2DCellKernels Simple2DCell Kernels
 * @{
 * \brief CUDA kernels and callers for the Simple2DCell class
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

/** @} */ //end of group declaration

#endif

