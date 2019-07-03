#define ENABLE_CUDA

#include "dynamicalFeatures.h"
#include "functions.h"
/*! \file dynamicalFeatures.cpp */

dynamicalFeatures::dynamicalFeatures(GPUArray<Dscalar2> &initialPos, BoxPtr _bx, Dscalar fractionAnalyzed)
    {
    Box = _bx;
    copyGPUArrayData(initialPos,iPos);
    N = iPos.size();
    if(fractionAnalyzed < 1)
        N = floor(N*fractionAnalyzed);
    };

Dscalar dynamicalFeatures::computeMSD(GPUArray<Dscalar2> &currentPos)
    {
    Dscalar msd = 0.0;
    ArrayHandle<Dscalar2> fPos(currentPos,access_location::host,access_mode::read);
    Dscalar2 disp,cur,init;
    for (int ii = 0; ii < N; ++ii)
        {
        cur = fPos.data[ii];
        init = iPos[ii];
        Box->minDist(init,cur,disp);
        msd += dot(disp,disp);
        };
    msd = msd / N;
    return msd;
    };

Dscalar dynamicalFeatures::computeOverlapFunction(GPUArray<Dscalar2> &currentPos, Dscalar cutoff)
    {
    Dscalar overlap = 0.0;
    ArrayHandle<Dscalar2> fPos(currentPos,access_location::host,access_mode::read);
    Dscalar2 disp,cur,init;
    for (int ii = 0; ii < N; ++ii)
        {
        cur = fPos.data[ii];
        init = iPos[ii];
        Box->minDist(init,cur,disp);
        if(norm(disp) < cutoff)
            overlap += 1;
        };
    overlap = overlap / N;
    return overlap;
    }
