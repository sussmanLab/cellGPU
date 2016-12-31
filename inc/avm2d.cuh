#ifndef __AVM2D_CUH__
#define __AVM2D_CUH__


#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"

//!Initialize the GPU's random number generator
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

#endif
