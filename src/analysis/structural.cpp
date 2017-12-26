#define ENABLE_CUDA

#include "structural.h"
/*! \file structural.cpp */

void structuralFeatures::computeRadialDistributionFunction(vector<Dscalar2> &points,vector<Dscalar2> &GofR, Dscalar binWidth)
    {
    int N = points.size();
    Dscalar L,b2,b3,b4;
    Box->getBoxDims(L,b2,b3,b4);

    //Initialize the answer vector
    int totalBins = floor(0.5*L/binWidth);
    GofR.resize(totalBins);
    for (int bb = 0; bb < totalBins; ++bb)
        GofR[bb] = make_Dscalar2((bb+0.5)*binWidth,0.0);

    //loop through points
    Dscalar2 dist;
    for (int ii = 0; ii < N-1; ++ii)
        {
        for (int jj = ii+1; jj < N; ++jj)
            {
            Box->minDist(points[ii],points[jj],dist);
            Dscalar d=norm(dist);
            int ibin = floor(d/binWidth);
            GofR[ibin].y += 1.0;
            };
        };
    //finally, normalize the function appropriately
    for (int bb = 0; bb < totalBins; ++bb)
        {
        Dscalar annulusArea = PI*(((bb+1)*binWidth)*((bb+1)*binWidth)-(bb*binWidth)*(bb*binWidth));
        Dscalar yVal = (2.0*GofR[bb].y/N) / annulusArea;
        GofR[bb].y=yVal;
        };
    };
