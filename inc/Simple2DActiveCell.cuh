#ifndef __SIMPLE2DACTIVECELL_CUH__
#define __SIMPLE2DACTIVECELL_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file Simple2DActiveCell.cuh
A file providing an interface to the relevant cuda calls for the Simple2DActiveCell class
*/

/** @defgroup Simple2DActiveCellKernels Simple2DActiveCell Kernels
 * @{
 * \brief CUDA kernels and callers for the Simple2DActiveCell class
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );
/** @} */ //end of group declaration


 #endif

