#ifndef __SELFPROPELLEDALIGNINGPARTICLEDYNAMICS_CUH__
#define __SELFPROPELLEDALIGNINGPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledAligningParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the selfPropelledAligningParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!set the vector of displacements from forces and activity
bool gpu_spp_aligning_eom_integration(
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double2 *motility,
                    double *cellDirectors,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    int Timestep,
                    double mu,
                    double J);

/** @} */ //end of group declaration
 #endif
