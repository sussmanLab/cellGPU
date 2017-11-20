#ifndef NoseHooverChainNVT_CUH
#define NoseHooverChainNVT_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*!
    \file NoseHooverChainNVT.cuh
This file provides an interface to cuda calls for integrating the NoseHooverChainNVT class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!Rescale the velocities according to the given scale factor
bool gpu_NoseHooverChainNVT_scale_velocities(
                    Dscalar2 *velocities,
                    Dscalar  *kineticEnergyScaleFactor,
                    int       N);



/** @} */ //end of group declaration
 #endif

