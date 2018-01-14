#ifndef __simpleVertexModelBase_CUH__
#define __simpleVertexModelBase_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "functions.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file simpleVertexModelBase.cuh
A file providing an interface to the relevant cuda calls for 2D vertex models. Takes care of functions
that don't depend on whether every vertex is three-fold coordinated or not.
*/
/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

bool gpu_vm_get_cell_positions(
                    Dscalar2 *d_cellPositions,
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &cellNeighborIndexer,
                    gpubox   &Box);

/** @} */ //end of group declaration
#endif
