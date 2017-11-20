#ifndef __SIMPLE2DCELL_CUH__
#define __SIMPLE2DCELL_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "gpubox.h"

/*!
 \file Simple2DCell.cuh
A file providing an interface to the relevant cuda calls for the Simple2DCell class
*/

/** @defgroup Simple2DCellKernels Simple2DCell Kernels
 * @{
 * \brief CUDA kernels and callers for the Simple2DCell class

 One might think that a "computeGeometry" function should be here, but this function depends
 too much on whether the primary degrees of freedom are cells or vertices
 */

//!Move degrees of freedom according to a set of displacements, and put them back in the unit cell
bool gpu_move_degrees_of_freedom(Dscalar2 *d_points,
                    Dscalar2 *d_disp,
                    int N,
                    gpubox &Box
                    );

//!The same as the above, but scale the displacements by a scalar (i.e., x[i] += scale*disp[i]
bool gpu_move_degrees_of_freedom(Dscalar2 *d_points,
                    Dscalar2 *d_disp,
                    Dscalar  scale,
                    int N,
                    gpubox &Box
                    );

//!A utility function; set all copmonents of an integer array to value
bool gpu_set_integer_array(int *d_array,
                           int value,
                           int N
                           );

/** @} */ //end of group declaration

#endif
