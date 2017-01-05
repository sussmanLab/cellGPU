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
                    int      *d_cvn,
                    int      *d_cv,
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
                    Dscalar2 *d_vertexPositions,
                    Dscalar2 *d_vertexForces,
                    Dscalar  *d_cellDirectors,
                    int      *d_vertexCellNeighbors,
                    curandState *d_curandRNGs,
                    Dscalar v0,
                    Dscalar Dr,
                    Dscalar deltaT,
                    gpubox &Box,
                    int Nvertices,
                    int Ncells);

bool gpu_avm_test_edges_for_T1(
                    Dscalar2 *d_v,
                    int      *d_vn,
                    int      *d_vflip,
                    int      *d_vcn,
                    int      *d_cvn,
                    int      *d_cv,
                    gpubox   &Box,
                    Dscalar  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow,
                    Index2D  &n_idx);

bool gpu_avm_flip_edges(
                    int      *d_vflip,
                    Dscalar2 *d_v,
                    int      *d_vn,
                    int      *d_vcn,
                    int      *d_cvn,
                    int      *d_cv,
                    gpubox   &Box,
                    Index2D  &n_idx, 
                    int      Nvertices);


bool gpu_avm_get_cell_positions(
                    Dscalar2 *d_p,
                    Dscalar2 *d_v,
                    int      *d_nn,
                    int      *d_n,
                    int      N, 
                    Index2D  &n_idx, 
                    gpubox   &Box);


#endif
