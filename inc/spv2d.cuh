#ifndef __SPV2D_CUH__
#define __SPV2D_CUH__


#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"


/*
bool gpu_init_curand(curandState *states,
                    unsigned long seed,
                    int N
                    );
*/


bool gpu_displace_and_rotate(
                    float2 *d_points,
                    float2 *d_force,
                    float  *directors,
                    float2 *d_motility,
                    int N,
                    float dt,
                    int seed,
//                    curandState *states,
                    gpubox &Box
                    );

bool gpu_compute_geometry(
                    float2 *d_points,
                    float2 *d_AP,
                    float2 *d_voro,
                    int    *d_nn,
                    int    *d_n,
                    int    N,
                    Index2D &n_idx,
                    gpubox &Box
                    );

bool gpu_force_sets(
                    float2 *d_points,
                    int    *d_nn,
                    float2 *d_AP,
                    float2 *d_APpref,
                    int4   *d_delSets,
                    int    *d_detOther,
                    float2 *d_forceSets,
                    float  KA,
                    float  KP,
                    int    N,
                    int    neighMax,
                    Index2D &n_idx,
                    gpubox &Box
                    );

bool gpu_force_sets_tensions(
                    float2 *d_points,
                    int    *d_nn,
                    float2 *d_AP,
                    float2 *d_APpref,
                    int4   *d_delSets,
                    int    *d_detOther,
                    float2 *d_forceSets,
                    int    *d_cellTypes,
                    float  KA,
                    float  KP,
                    float  gamma,
                    int    N,
                    int    neighMax,
                    Index2D &n_idx,
                    gpubox &Box
                    );

bool gpu_sum_force_sets(
                    float2 *d_forceSets,
                    float2 *d_forces,
                    int    *d_nn,
                    int     N,
                    Index2D &n_idx
                    );
#endif

