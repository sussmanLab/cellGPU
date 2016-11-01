#ifndef __DELAUNAYMD_CUH__
#define __DELAUNAUMD_CUH__


#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"



using namespace voroguppy;



bool gpu_move_particles(float2 *d_points,
                    float2 *d_disp,
                    int N,
                    gpubox &Box
                    );



#endif

