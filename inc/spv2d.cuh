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
                    int N,
                    float dt,
                    float Dr,
                    float v0,
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


#endif

