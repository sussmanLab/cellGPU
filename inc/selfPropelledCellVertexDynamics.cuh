#ifndef __SELFPROPELLEDCELLVERTEXDYNAMICS_CUH__
#define __SELFPROPELLEDCELLVERTEXDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledCellVertexDynamics.cuh
A file providing an interface to the relevant cuda calls for the selfPropelledCellVertex class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!set the vector of displacements from forces and activity
bool gpu_spp_cellVertex_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *displacements,
                    Dscalar2 *motility,
                    Dscalar  *cellDirectors,
                    int      *vertexNeighbors,
                    curandState *RNGs,
                    int Nvertices,
                    int Ncells,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu);

/** @} */ //end of group declaration
 #endif

