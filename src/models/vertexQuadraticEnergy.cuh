#ifndef __vertexQuadraticEnergy_CUH__
#define __vertexQuadraticEnergy_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "functions.h"
#include "indexer.h"
#include "periodicBoundaries.h"

/*!
 \file
A file providing an interface to the relevant cuda calls for the VertexQuadraticEnergy class
*/

/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

bool gpu_avm_force_sets(
                    int      *d_vertexCellNeighbors,
                    double2 *d_voroCur,
                    double4 *d_voroLastNext,
                    double2 *d_AreaPerimeter,
                    double2 *d_AreaPerimeterPreferences,
                    double2 *d_vertexForceSets,
                    int nForceSets,
                    double KA, double KP);

bool gpu_avm_sum_force_sets(
                    double2 *d_vertexForceSets,
                    double2 *d_vertexForces,
                    int      Nvertices);

/** @} */ //end of group declaration
#endif
