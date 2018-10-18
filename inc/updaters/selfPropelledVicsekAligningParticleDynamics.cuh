#ifndef __SELFPROPELLEDVICSEKALIGNINGPARTICLEDYNAMICS_CUH__
#define __SELFPROPELLEDVICSEKALIGNINGPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include "indexer.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledVicsekAligningParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the
selfPropelledVicsekAligningParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!set the vector of displacements from forces and activity
bool gpu_spp_vicsek_aligning_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *velocities,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar *cellDirectors,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu,
                    Dscalar Eta);

/** @} */ //end of group declaration
 #endif
