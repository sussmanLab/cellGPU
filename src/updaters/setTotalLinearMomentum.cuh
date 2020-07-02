#ifndef setTotalLinearMomentum_CUH
#define setTotalLinearMomentum_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*!
    \file setTotalLinearMomentum.cuh 
This file provides an interface to cuda calls for setting the total linear momentum of the system
*/

/** @addtogroup updatersKernels updaters Kernels
 * @{
 */

//! shift the velocities to get the target linear momentum
bool gpu_shift_momentum(Dscalar2   *velocities,
                        Dscalar *masses,
                        Dscalar2 pShift,
                        int N);

/** @} */ //end of group declaration
 #endif

