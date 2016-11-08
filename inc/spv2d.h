//spv.h
#ifndef SPV_H
#define SPV_H

using namespace std;

#include <stdio.h>

#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"


#include "DelaunayMD.h"

class SPV2D : public DelaunayMD
    {
    private:
        GPUArray<float2> points;      //vector of particle positions


    public:
        SPV2D(int n);

    };


#endif
