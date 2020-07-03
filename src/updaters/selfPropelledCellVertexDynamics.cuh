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
                    double2 *forces,
                    double2 *displacements,
                    double2 *motility,
                    double  *cellDirectors,
                    int      *vertexNeighbors,
                    curandState *RNGs,
                    int Nvertices,
                    int Ncells,
                    double deltaT,
                    int Timestep,
                    double mu);

/** @} */ //end of group declaration
 #endif

