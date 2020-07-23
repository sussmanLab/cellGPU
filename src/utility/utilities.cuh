#ifndef utilities_CUH__
#define utilities_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "gpuarray.h"
/*!
 \file utilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//! (double) ans = (double2) vec1 . vec2
bool gpu_dot_double2_vectors(double2 *d_vec1,
                              double2 *d_vec2,
                              double  *d_ans,
                              int N);

//!A trivial reduction of an array by one thread in serial. Think before you use this.
bool gpu_serial_reduction(
                    double *array,
                    double *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm.
bool gpu_parallel_reduction(
                    double *input,
                    double *intermediate,
                    double *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm for double2 arrays.
bool gpu_parallel_reduction(
                    double2 *input,
                    double2 *intermediate,
                    double2 *output,
                    int helperIdx,
                    int N);

//! (double2) ans = (double2) vec1 * vec2
bool gpu_dot_double_double2_vectors(double *d_vec1,
                              double2 *d_vec2,
                              double2  *d_ans,
                              int N);

//!set every element of an array to the specified value
template<typename T>
bool gpu_set_array(T *arr,
                   T value,
                   int N,
                   int maxBlockSize=512);

//! answer = answer+adder
template<typename T>
bool gpu_add_gpuarray(GPUArray<T> &answer,
                       GPUArray<T> &adder,
                       int N,
                       int block_size=512);

//!copy data into target on the device...copies the first Ntotal elements into the target array, by default it copies all elements
template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,
                       GPUArray<T> &copyFrom,
                       int numberOfElementsToCopy = -1,
                       int block_size=512);
/** @} */ //end of group declaration
#endif
