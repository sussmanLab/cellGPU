#ifndef __AVM2D_CUH__
#define __AVM2D_CUH__


#include "std_include.h"
#include <cuda_runtime.h>

#include "cu_functions.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file avm.cuh
A file providing an interface to the relevant cuda calls for the AVM2D class
*/

/** @defgroup avmKernels AVM Kernels
 * @{
 * \brief CUDA kernels for the AVM2D class
 */

//!Initialize the GPU's random number generator
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

bool gpu_avm_geometry(
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

bool gpu_avm_force_sets(
                    int      *d_vertexCellNeighbors,
                    Dscalar2 *d_voroCur,
                    Dscalar4 *d_voroLastNext,
                    Dscalar2 *d_AreaPerimeter,
                    Dscalar2 *d_AreaPerimeterPreferences,
                    Dscalar2 *d_vertexForceSets,
                    int nForceSets,
                    Dscalar KA, Dscalar KP);

bool gpu_avm_sum_force_sets(
                    Dscalar2 *d_vertexForceSets,
                    Dscalar2 *d_vertexForces,
                    int      Nvertices);

bool gpu_avm_displace_and_rotate(
                    Dscalar2 *d_vertexPositions,
                    Dscalar2 *d_vertexForces,
                    Dscalar  *d_cellDirectors,
                    int      *d_vertexCellNeighbors,
                    curandState *d_curandRNGs,
                    Dscalar v0,
                    Dscalar Dr,
                    Dscalar deltaT,
                    gpubox &Box,
                    int Nvertices,
                    int Ncells);

bool gpu_avm_test_edges_for_T1(
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
                    Index2D  &n_idx);

bool gpu_avm_flip_edges(
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexEdgeFlipsCurrent,
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_finishedFlippingEdges,
                    Dscalar  T1Threshold,
                    gpubox   &Box,
                    Index2D  &n_idx,
                    int      Nvertices,
                    int      Ncells);

bool gpu_avm_get_cell_positions(
                    Dscalar2 *d_cellPositions,
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box);

/** @} */ //end of group declaration
#endif
