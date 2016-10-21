#ifndef __DTEST_CUH__
#define __DTEST_CUH__


#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"



using namespace voroguppy;



//extern "C"
//{
bool gpu_test_circumcircles(bool *d_redo,
                                  int *d_ccs,
                                  float2 *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int Nmax,
                                  int xsize,
                                  int ysize,
                                  float boxsize,
                                  gpubox &Box,
                                  Index2D &ci,
                                  Index2D &cli
                                  );
//}
#endif

