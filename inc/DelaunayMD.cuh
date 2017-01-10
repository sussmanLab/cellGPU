#ifndef __DELAUNAYMD_CUH__
#define __DELAUNAYMD_CUH__


#include <cuda_runtime.h>
#include "std_include.h"
#include "indexer.h"
#include "gpubox.h"

/*!
 \file DelaunayMD.cuh
A file providing an interface to the relevant cuda calls for the DelaunayMD class
*/

/** @defgroup DelaunayMDKernels DelaunayMD Kernels
 * @{
 * \brief CUDA kernels for the DelaunayMD class
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
                            int &fail
                            );

//!Move particles according to a set of displacements, and put them back in the unit cell
bool gpu_move_particles(Dscalar2 *d_points,
                    Dscalar2 *d_disp,
                    int N,
                    gpubox &Box
                    );
/** @} */ //end of group declaration

#endif

