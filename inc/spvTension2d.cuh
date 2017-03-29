#ifndef __SPVTENSION2D_CUH__
#define __SPVTENSION2D_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"
#include "spv2d.cuh"

/*!
 \file
A file providing an interface to the relevant cuda calls for the SPV2D class
*/

/** @defgroup spvKernels SPV Kernels
 * @{
 * \brief CUDA kernels and callers for the SPV2D class
 */

//!Compute the contribution to the net force on vertex i from each of i's voronoi vertices
bool gpu_spvSimpleTension_force_sets(
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
