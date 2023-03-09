#ifndef __voronoiModelBase_CUH__
#define __voronoiModelBase_CUH__

#include <cuda_runtime.h>
#include "std_include.h"
#include "indexer.h"
#include "periodicBoundaries.h"

/*!
 \file voronoiModelBase.cuh
A file providing an interface to the relevant cuda calls for the base voronoi model class
*/

/** @defgroup voronoiModelBaseKernels voronoiModelBase Kernels
 * @{
 * \brief CUDA kernels and callers for the voronoiModelBase class
 */

//!update NeighIdx data structure on the gpu
bool gpu_update_neighIdxs(int *neighborNum,
                          int *neighNumScan,
                          int2 *neighIdxs,
                          int &NeighIdxNum,
                          int Ncells);
//!update delSets structures on the GPU
bool gpu_all_del_sets(int *neighborNum,
                      int *neighbors,
                      int2 *delSets,
                      int * delOther,
                      int Ncells,
                      Index2D &nIdx);

//!Test an array of circumcenters for the empty-circumcircle property
bool gpu_test_circumcenters(
                            int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            double2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            double boxsize,
                            periodicBoundaries &Box,
                            Index2D &ci,
                            Index2D &cli,
                            int *fail
                            );

//!compute the area and perimeter of all Voronoi cells, and save the voronoi vertices
bool gpu_compute_voronoi_geometry(
                    const double2 *d_points,
                    double2 *d_AP,
                    const int    *d_nn,
                    const int    *d_n,
                    double2 *d_vc,
                    double4 *d_vln,
                    int    N,
                    Index2D &n_idx,
                    periodicBoundaries &Box,
                    bool useGPU = true,
                    int nThreads = 1
                    );

/** @} */ //end of group declaration

#endif
