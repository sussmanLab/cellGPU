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
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double2 *motility,
                    double *cellDirectors,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double deltaT,
                    int Timestep,
                    double mu,
                    double Eta);

/** @} */ //end of group declaration
 #endif
