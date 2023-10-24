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

void dynamicalFeatures::setCageNeighbors(GPUArray<int> &neighbors, GPUArray<int> &neighborNum, Index2D n_idx)
    {
    nIdx = Index2D(n_idx.getW(),n_idx.getH());
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);

    cageNeighbors.resize(N);
    for (int ii = 0; ii < N; ++ii)
        {
        int neighs = h_nn.data[ii];
        vector<int> ns(neighs);
        for (int nn = 0; nn < neighs; ++nn)
            {
            ns[nn]=h_n.data[nIdx(nn,ii)];
            };
        cageNeighbors[ii] = ns;
        }
    };

void dynamicalFeatures::computeCageRelativeDisplacements(GPUArray<double2> &currentPos)
    {
    currentDisplacements.resize(N);
    cageRelativeDisplacements.resize(N);
    //first, compute the vector of current displacements
    ArrayHandle<double2> fPos(currentPos,access_location::host,access_mode::read);
    double2 disp,cur,init;
    for (int ii = 0; ii < N; ++ii)
        {
        cur = fPos.data[ii];
        init = iPos[ii];
        Box->minDist(init,cur,disp);
        currentDisplacements[ii] = disp;
        }
    //now, for each particle, compute the cage-relativedisplacement
    for(int ii = 0; ii < N; ++ii)
        {
        //self term
        cur = currentDisplacements[ii];
        //subtract off net neighbor motion
        int nNeighs = cageNeighbors[ii].size();
        for(int nn = 0; nn < nNeighs; ++nn)
            {
            int neighborIndex = nIdx(nn,ii);
            cur = cur - ((1./(double)nNeighs)) * currentDisplacements[neighborIndex];
            }
        cageRelativeDisplacements[ii] = cur;
        };
    };

double dynamicalFeatures::computeCageRelativeMSD(GPUArray<double2> &currentPos)
    {
    //call helper function to compute the vector of current cage-relative displacement vectors
    computeCageRelativeDisplacements(currentPos);

    //then just compute the MSD of that set of vectors..

    double2 disp;
    double msd = 0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        disp = cageRelativeDisplacements[ii];
        msd += dot(disp,disp);
        };
    msd = msd / N;
    return msd;
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
