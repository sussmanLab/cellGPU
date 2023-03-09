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

//! (double) ans = (double2) vec1 . vec2
bool gpu_prepare_KE_vector(double2   *velocities,
                              double *masses,
                              double *keArray,
                              int N);

//!Rescale the velocities according to the given scale factor
bool gpu_NoseHooverChainNVT_scale_velocities(
                    double2 *velocities,
                    double  *kineticEnergyScaleFactor,
                    int       N);

//!update the velocities according to the forces and the masses
bool gpu_NoseHooverChainNVT_update_velocities(
                    double2 *velocities,
                    double2 *forces,
                    double  *masses,
                    double  deltaT,
                    int       N);

/** @} */ //end of group declaration
 #endif

