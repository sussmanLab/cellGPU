#ifndef __vertexModelBase_CUH__
#define __vertexModelBase_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "functions.h"
#include "indexer.h"
#include "periodicBoundaries.h"

/*!
 \file
A file providing an interface to the relevant cuda calls for 2D vertex models
*/
/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

bool gpu_vm_geometry(
                    double2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_vertexCellNeighbors,
                    double2 *d_voroCur,
                    double4  *d_voroLastNext,
                    double2 *d_AreaPeri,
                    int      N,
                    Index2D  &n_idx,
                    periodicBoundaries   &Box);

bool gpu_vm_get_cell_centroids(
                    double2 *d_cellPositions,
                    double2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    periodicBoundaries   &Box);

bool gpu_vm_get_cell_positions(
                    double2 *d_cellPositions,
                    double2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    periodicBoundaries   &Box);

bool gpu_vm_test_edges_for_T1(
                    double2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    periodicBoundaries   &Box,
                    double  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow,
                    Index2D  &n_idx);

bool gpu_vm_parse_multiple_flips(
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexEdgeFlipsCurrent,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_finishedFlippingEdges,
                    int      *d_edgeFlips,
                    int4     *d_cellSets,
                    Index2D  &n_idx,
                    int      Ncells);

bool gpu_vm_flip_edges(
                    int      *d_vertexEdgeFlipsCurrent,
                    double2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_edgeFlips,
                    int4     *d_cellSets,
                    periodicBoundaries   &Box,
                    Index2D  &n_idx,
                    int      Nvertices,
                    int      Ncells);

/** @} */ //end of group declaration
#endif
