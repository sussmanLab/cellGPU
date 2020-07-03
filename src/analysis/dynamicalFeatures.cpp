#include "dynamicalFeatures.h"
#include "functions.h"
/*! \file dynamicalFeatures.cpp */

dynamicalFeatures::dynamicalFeatures(GPUArray<double2> &initialPos, PeriodicBoxPtr _bx, double fractionAnalyzed)
    {
    Box = _bx;
    copyGPUArrayData(initialPos,iPos);
    N = iPos.size();
    if(fractionAnalyzed < 1)
        N = floor(N*fractionAnalyzed);
    };

double dynamicalFeatures::computeMSD(GPUArray<double2> &currentPos)
    {
    double msd = 0.0;
    ArrayHandle<double2> fPos(currentPos,access_location::host,access_mode::read);
    double2 disp,cur,init;
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

double dynamicalFeatures::computeOverlapFunction(GPUArray<double2> &currentPos, double cutoff)
    {
    double overlap = 0.0;
    ArrayHandle<double2> fPos(currentPos,access_location::host,access_mode::read);
    double2 disp,cur,init;
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
