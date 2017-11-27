#ifndef utilities_CUH__
#define utilities_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
/*!
 \file utilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//! (Dscalar) ans = (Dscalar2) vec1 . vec2
bool gpu_dot_Dscalar2_vectors(Dscalar2 *d_vec1,
                              Dscalar2 *d_vec2,
                              Dscalar  *d_ans,
                              int N);

//!A trivial reduction of an array by one thread in serial. Think before you use this.
bool gpu_serial_reduction(
                    Dscalar *array,
                    Dscalar *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm.
bool gpu_parallel_reduction(
                    Dscalar *input,
                    Dscalar *intermediate,
                    Dscalar *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm for Dscalar2 arrays.
bool gpu_parallel_reduction(
                    Dscalar2 *input,
                    Dscalar2 *intermediate,
                    Dscalar2 *output,
                    int helperIdx,
                    int N);

//! (Dscalar2) ans = (Dscalar2) vec1 * vec2
bool gpu_dot_Dscalar_Dscalar2_vectors(Dscalar *d_vec1,
                              Dscalar2 *d_vec2,
                              Dscalar2  *d_ans,
                              int N);

/** @} */ //end of group declaration
#endif
