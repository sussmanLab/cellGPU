#ifndef __VoronoiTENSION2D_CUH__
#define __VoronoiTENSION2D_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"
#include "voronoiQuadraticEnergy.cuh"

/*!
 \file
A file providing an interface to the relevant cuda calls for the Voronoi2D class
*/

/** @defgroup spvKernels SPV Kernels
 * @{
 * \brief CUDA kernels and callers for the Voronoi2D class
 */

//!Compute the contribution to the net force on vertex i from each of i's voronoi vertices with general tensions
bool gpu_VoronoiTension_force_sets(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2    *d_nidx,
                    int     *d_cellTypes,
                    Dscalar *d_tensionMatrix,
                    Index2D &cellTypeIndexer,
                    Dscalar  KA,
                    Dscalar  KP,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    );

//!Compute the contribution to the net force on vertex i from each of i's voronoi vertices
bool gpu_VoronoiSimpleTension_force_sets(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2    *d_nidx,
                    int     *d_cellTypes,
                    Dscalar  KA,
                    Dscalar  KP,
                    Dscalar  gamma,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    );

/** @} */ //end of group declaration
#endif
