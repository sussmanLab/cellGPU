#ifndef __LANGEVINDYNAMICS_CUH__
#define __LANGEVINYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file langevinDynamics.cuh
A file providing an interface to the relevant cuda calls for the brownianParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!implement the combined "B" and "O" steps (see DOI references in langevinDynamics.h file)
bool gpu_langevin_BandO_operation(
                    double2 *velocities,
                    double2 *forces,
                    double2 *displacements,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    double gamma,
                    double T);

/** @} */ //end of group declaration
 #endif
