#ifndef __vertexModelBase_CUH__
#define __vertexModelBase_CUH__

#include "simpleVertexModelBase.cuh"
#include "std_include.h"
#include <cuda_runtime.h>
#include "functions.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file vertexModelBase.cuh
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
                    Index2D  &cellNeighborIndexer,
                    gpubox   &Box);

bool gpu_vm_test_edges_for_T1(
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    gpubox   &Box,
                    Dscalar  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow,
                    Index2D  &cellNeighborIndexer);

bool gpu_vm_parse_multiple_flips(
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexEdgeFlipsCurrent,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_finishedFlippingEdges,
                    Index2D  &cellNeighborIndexer,
                    int      Ncells);

bool gpu_vm_flip_edges(
                    int      *d_vertexEdgeFlipsCurrent,
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    Dscalar  T1Threshold,
                    gpubox   &Box,
                    Index2D  &cellNeighborIndexer,
                    int      Nvertices,
                    int      Ncells);

/** @} */ //end of group declaration
#endif
