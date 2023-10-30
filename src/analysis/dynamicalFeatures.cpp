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

void dynamicalFeatures::computeDisplacements(GPUArray<double2> &currentPos)
    {
    currentDisplacements.resize(N);
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
    };

void dynamicalFeatures::computeCageRelativeDisplacements(GPUArray<double2> &currentPos)
    {
    cageRelativeDisplacements.resize(N);
    //first, compute the vector of current displacements
    computeDisplacements(currentPos);

    //now, for each particle, compute the cage-relativedisplacement
    double2 cur;
    double2 temp;
    for(int ii = 0; ii < N; ++ii)
        {
        //self term
        cur = currentDisplacements[ii];
        //subtract off net neighbor motion
        int nNeighs = cageNeighbors[ii].size();
        temp.x=0;temp.y=0;
        for(int nn = 0; nn < nNeighs; ++nn)
            {
            int neighborIndex = cageNeighbors[ii][nn];
            temp.x = temp.x + currentDisplacements[neighborIndex].x/nNeighs;
            temp.y = temp.y + currentDisplacements[neighborIndex].y/nNeighs;
            }
        cageRelativeDisplacements[ii].x = cur.x - (1./((double) nNeighs))* temp.x;
        cageRelativeDisplacements[ii].y = cur.y - (1./((double) nNeighs))* temp.y;
        };
    };

double dynamicalFeatures::MSDhelper(vector<double2> &displacements)
    {
    double2 disp;
    double msd = 0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        disp = displacements[ii];
        msd += dot(disp,disp);
        };
    msd = msd / N;
    return msd;
    };

double dynamicalFeatures::computeMSD(GPUArray<double2> &currentPos)
    {
    //call helper function to compute vector of current displacements
    computeDisplacements(currentPos);
    double result = MSDhelper(currentDisplacements);
    return result;
    };

double dynamicalFeatures::computeCageRelativeMSD(GPUArray<double2> &currentPos)
    {
    //call helper function to compute the vector of current cage-relative displacement vectors
    computeCageRelativeDisplacements(currentPos);

    //then just compute the MSD of that set of vectors..
    double result = MSDhelper(cageRelativeDisplacements);
    return result;
    };

/*!
In d-dimensions, the contribution of the angularly average of <exp(I k.r)> is
(Power(2,-1 + n/2.)*(-(k*r*BesselJ(1 + n/2.,k*r)) + n*BesselJ(n/2.,k*r))*Gamma(n/2.))/
   Power(k*r,n/2.)
See: THE DEBYE SCATTERING FORMULA IN n DIMENSION, Wieder, J. Math. Comput. Sci. 2 (2012), No. 4, 1086-1090
*/
double dynamicalFeatures::angularAverageSISF(vector<double2> &displacements, double k)
    {
    double2 disp;
    double sisfContribution = 0.0;
    double kr;
    for (int ii = 0; ii < N; ++ii)
        {
        disp = displacements[ii];
        kr = k*sqrt(dot(disp,disp));
        sisfContribution += std::cyl_bessel_j((double) 0.0, (double) kr);
        };
    sisfContribution = sisfContribution / N;
    return sisfContribution;
    };

double dynamicalFeatures::computeSISF(GPUArray<double2> &currentPos, double k)
    {
    //call helper function to compute the vector of current cage-relative displacement vectors
    computeDisplacements(currentPos);

    //then just compute the MSD of that set of vectors..
    double result = angularAverageSISF(currentDisplacements,k);
    return result;
    };
/*
 Just compute cage relative displacements and pass that vector to the helper function
 */
double dynamicalFeatures::computeCageRelativeSISF(GPUArray<double2> &currentPos, double k)
    {
    //call helper function to compute the vector of current cage-relative displacement vectors
    computeCageRelativeDisplacements(currentPos);

    //then just compute the MSD of that set of vectors..
    double result = angularAverageSISF(cageRelativeDisplacements,k);
    return result;
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
