#ifndef __AVM2D_CUH__
#define __AVM2D_CUH__


#include "std_include.h"
#include <cuda_runtime.h>

#include "cu_functions.h"
#include "indexer.h"
#include "gpubox.h"

//!Initialize the GPU's random number generator
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

bool gpu_avm_geometry(
                    Dscalar2 *d_p,
                    Dscalar2 *d_v,
                    int      *d_nn,
                    int      *d_n,
                    int      *d_vcn,
                    Dscalar2 *d_vc,
                    Dscalar4  *d_vln,
                    Dscalar2 *d_AP,
                    int      N, 
                    Index2D  &n_idx, 
                    gpubox   &Box);

bool gpu_avm_force_sets(
                    int      *d_vcn,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    Dscalar2 *d_fs,
                    int nForceSets,
                    Dscalar KA, Dscalar KP);

bool gpu_avm_sum_force_sets(
                    Dscalar2 *d_fs,
                    Dscalar2 *d_f,
                    int      Nvertices);
                    
bool gpu_avm_displace_and_rotate(
                    Dscalar2 *d_v,
                    Dscalar2 *d_f,
                    Dscalar *d_vd,
                    curandState *d_cs,
                    Dscalar v0,
                    Dscalar Dr,
                    Dscalar deltaT,
                    int Timestep,
                    gpubox &Box,
                    int Nvertices);

#endif
