#ifndef __vertexModelBase_CUH__
#define __vertexModelBase_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "functions.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file
A file providing an interface to the relevant cuda calls for 2D vertex models
*/
/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

bool gpu_vm_geometry(
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_vertexCellNeighbors,
                    Dscalar2 *d_voroCur,
                    Dscalar4  *d_voroLastNext,
                    Dscalar2 *d_AreaPeri,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box);

bool gpu_vm_get_cell_positions(
                    Dscalar2 *d_cellPositions,
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box);

bool gpu_vm_displace(
                    Dscalar2 *d_vertexPositions,
                    Dscalar2 *d_vertexDisplacements,
                    gpubox &Box,
                    int Nvertices);

/** @} */ //end of group declaration

#endif
