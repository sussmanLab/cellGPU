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

//! (Dscalar) ans = (Dscalar2) vec1 . vec2
bool gpu_prepare_KE_vector(Dscalar2   *velocities,
                              Dscalar *masses,
                              Dscalar *keArray,
                              int N);

//!Rescale the velocities according to the given scale factor
bool gpu_NoseHooverChainNVT_scale_velocities(
                    Dscalar2 *velocities,
                    Dscalar  *kineticEnergyScaleFactor,
                    int       N);

//!update the velocities according to the forces and the masses
bool gpu_NoseHooverChainNVT_update_velocities(
                    Dscalar2 *velocities,
                    Dscalar2 *forces,
                    Dscalar  *masses,
                    Dscalar  deltaT,
                    int       N);

/** @} */ //end of group declaration
 #endif

