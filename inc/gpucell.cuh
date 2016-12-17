#ifndef __GPUCELL_CUH__
#define __GPUCELL_CUH__


#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"

bool gpu_compute_cell_list(Dscalar2 *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int &Nmax,
                                  int xsize,
                                  int ysize,
                                  Dscalar boxsize,
                                  gpubox &Box,
                                  Index2D &ci,
                                  Index2D &cli,
                                  int *d_assist
                                  );

bool gpu_zero_array(int *arr,
                    int N
                    );
bool gpu_zero_array(unsigned int *arr,
                    int      N
                    );


#endif

