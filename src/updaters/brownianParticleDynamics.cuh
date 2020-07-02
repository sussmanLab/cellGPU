#ifndef __BROWNIANPARTICLEDYNAMICS_CUH__
#define __BROWNIANPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file brownianParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the brownianParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!set the vector of displacements from forces and noise
bool gpu_brownian_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *displacements,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    Dscalar mu,
                    Dscalar T);

/** @} */ //end of group declaration
 #endif
