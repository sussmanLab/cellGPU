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
                    Dscalar2 *forces,
                    Dscalar2 *velocities,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar *cellDirectors,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu,
                    Dscalar J);

/** @} */ //end of group declaration
 #endif
