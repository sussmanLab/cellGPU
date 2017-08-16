#ifndef __voronoiModelBase_CUH__
#define __voronoiModelBase_CUH__

#include <cuda_runtime.h>
#include "std_include.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file voronoiModelBase.cuh
A file providing an interface to the relevant cuda calls for the base voronoi model class
*/

/** @defgroup voronoiModelBaseKernels voronoiModelBase Kernels
 * @{
 * \brief CUDA kernels and callers for the voronoiModelBase class
 */

//!Test an array of circumcenters for the empty-circumcircle property
bool gpu_test_circumcenters(
                            int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            Dscalar2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            Dscalar boxsize,
                            gpubox &Box,
                            Index2D &ci,
                            Index2D &cli,
                            int *fail
                            );

//!compute the area and perimeter of all Voronoi cells, and save the voronoi vertices
bool gpu_compute_voronoi_geometry(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    int    *d_nn,
                    int    *d_n,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    int    N,
                    Index2D &n_idx,
                    gpubox &Box
                    );

/** @} */ //end of group declaration

#endif
