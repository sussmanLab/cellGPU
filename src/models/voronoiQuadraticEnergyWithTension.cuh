#ifndef __VoronoiTENSION2D_CUH__
#define __VoronoiTENSION2D_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "periodicBoundaries.h"
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
                    double2 *d_points,
                    double2 *d_AP,
                    double2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    double2 *d_vc,
                    double4 *d_vln,
                    double2 *d_forceSets,
                    int2    *d_nidx,
                    int     *d_cellTypes,
                    double *d_tensionMatrix,
                    Index2D &cellTypeIndexer,
                    double  KA,
                    double  KP,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    periodicBoundaries &Box
                    );

//!Compute the contribution to the net force on vertex i from each of i's voronoi vertices
bool gpu_VoronoiSimpleTension_force_sets(
                    double2 *d_points,
                    double2 *d_AP,
                    double2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    double2 *d_vc,
                    double4 *d_vln,
                    double2 *d_forceSets,
                    int2    *d_nidx,
                    int     *d_cellTypes,
                    double  KA,
                    double  KP,
                    double  gamma,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    periodicBoundaries &Box
                    );

/** @} */ //end of group declaration
#endif
