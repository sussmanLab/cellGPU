#ifndef __DELAUNAYMD_CUH__
#define __DELAUNAYMD_CUH__


#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"

bool gpu_test_circumcenters(
                            int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            float2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            float boxsize,
                            gpubox &Box,
                            Index2D &ci,
                            Index2D &cli,
                            int &fail
                            );


bool gpu_move_particles(float2 *d_points,
                    float2 *d_disp,
                    int N,
                    gpubox &Box
                    );

#endif

