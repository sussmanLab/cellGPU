#ifndef __voronoiModelBase_CUH__

#include <cuda_runtime.h>
#include "std_include.h"
#include "indexer.h"
#include "periodicBoundaries.h"


/*!
 \file DelaunayGPU.cuh
A file providing an interface to the relevant cuda calls for the delaunay GPU class
*/

/** @defgroup DelaunayGPUKernels DelaunayGPU Kernels
 * @{
 * \brief CUDA kernels and callers for the DelaunayGPU class
 */

//!construct the circumcircles int3 data structure from a triangulation
bool gpu_get_circumcircles(int *neighbors,
                           int *neighnum,
                           int3 *circumcircles,
                           int *assist,
                           int N,
                           Index2D &nIdx
                          );

//test the triangulation to see if it is still valid
bool gpu_test_circumcircles(int *d_repair,
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
                            bool GPUcompute,
                            unsigned int OMPThreadsNum
                            );
//!Find enclosing polygons to serve as candidate one-rings
bool gpu_voronoi_calc(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      Index2D GPU_idx,
                      bool GPUcompute,
                      unsigned int OMPThreadsNum
                      );

//!Find enclosing polygons to serve as candidate one-rings on the testAndRepair branch
bool gpu_voronoi_calc_no_sort(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      Index2D GPU_idx,
                      bool GPUcompute,
                      unsigned int OMPThreadsNum
                      );

//!Find the one-rings on the global triangulation branch of the algorithm
bool gpu_get_neighbors(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      Index2D GPU_idx,
                      int* maximumNeighborNum,
                      int currentMaxNeighborNum,
                      bool GPUcompute,
                      unsigned int OMPThreadsNum
                      );

//!Find the one-rings on the testAndRepair branch
bool gpu_get_neighbors_no_sort(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      Index2D GPU_idx,
                      int* maximumNeighborNum,
                      int currentMaxNeighborNum,
                      bool GPUcompute,
                      unsigned int OMPThreadsNum
                      );

/** @} */ //end of group declaration
#endif
