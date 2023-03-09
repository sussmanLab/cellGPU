#include "structuralFeatures.h"
/*! \file structuralFeatures.cpp */

/*!
A brute-force, O(N^2) computation of the radial distribution function for the point pattern. The
answer is stored in the GofR vector.
*/
void structuralFeatures::computeRadialDistributionFunction(vector<double2> &points,vector<double2> &GofR, double binWidth)
    {
    int N = points.size();
    double L,b2,b3,b4;
    Box->getBoxDims(L,b2,b3,b4);

    //Initialize the answer vector
    int totalBins = floor(0.5*L/binWidth);
    GofR.resize(totalBins);
    for (int bb = 0; bb < totalBins; ++bb)
        GofR[bb] = make_double2((bb+0.5)*binWidth,0.0);

    //loop through points
    double2 dist;
    for (int ii = 0; ii < N-1; ++ii)
        {
        for (int jj = ii+1; jj < N; ++jj)
            {
            Box->minDist(points[ii],points[jj],dist);
            double d=norm(dist);
            int ibin = floor(d/binWidth);
            if (ibin < totalBins)
                GofR[ibin].y += 1.0;
            };
        };
    //finally, normalize the function appropriately
    for (int bb = 0; bb < totalBins; ++bb)
        {
        double annulusArea = PI*(((bb+1)*binWidth)*((bb+1)*binWidth)-(bb*binWidth)*(bb*binWidth));
        double yVal = (2.0*GofR[bb].y/N) / annulusArea;
        GofR[bb].y=yVal;
        };
    };

/*!
A calculation of the (isotropic) structure factor for the 2D point pattern in "points"
The calculation is based on making a grid of \rho(K) at a lattice of K points (whose maximum value
is given by (2 Pi / L)*floor(L)*intKMax), computing S(k), and then averaging that S(K)
*/
void structuralFeatures::computeStructureFactor(vector<double2> &points,vector<double2> &SofK, double intKMax,double dk)
    {
    int N = points.size();
    double L,b2,b3,b4;
    Box->getBoxDims(L,b2,b3,b4);
    double deltaK = 2*PI/L;

    //Initialize the lattice of wavevectors
    int maxLatticeInt = floor(L*intKMax);
    vector<double2> zeroVector(maxLatticeInt,make_double2(0.0,0.0));
    vector<double> zeroVector1(maxLatticeInt,0.0);
    vector<vector<double2> > wavevectors(maxLatticeInt,zeroVector);
    for (int ii = 0; ii < maxLatticeInt; ++ii)
        for (int jj = 0; jj < maxLatticeInt; ++jj)
            {
            wavevectors[ii][jj].x = (ii)*deltaK;
            wavevectors[ii][jj].y = (jj)*deltaK;
            };
    //evaluate the transformed density at each lattice vector
    vector<vector<double2> > rhoK(maxLatticeInt,zeroVector);
    for (int nn = 0; nn < N; ++nn)
        for (int ii = 0; ii < maxLatticeInt; ++ii)
            for (int jj = 0; jj < maxLatticeInt; ++jj)
                {
                double2 K = wavevectors[ii][jj];
                double argument= dot(K,points[nn]);
                rhoK[ii][jj].x += cos(argument);
                rhoK[ii][jj].y += sin(argument);
                };
    //evaluate S(k), assuming isotropy
    vector<vector<double> > SK(maxLatticeInt,zeroVector1);
    for (int ii = 0; ii < maxLatticeInt; ++ii)
        for (int jj = 0; jj < maxLatticeInt; ++jj)
            {
            double2 rk = rhoK[ii][jj];
            SK[ii][jj] = (rk.x*rk.x+rk.y*rk.y)/N;
            };

    //finally, average the grid points to the answer
    vector<double2> answer;
    double binWidth = deltaK*dk;
    double kmax = wavevectors[maxLatticeInt-1][maxLatticeInt-1].x;
    for (double rmin = deltaK-0.5*binWidth; rmin< kmax-binWidth; rmin +=binWidth)
        {
        double rmax = rmin + binWidth;
        double value = 0.0;
        int inSum = 0;
        for (int ii = 0; ii < maxLatticeInt; ++ii)
            {
            for (int jj = 0; jj < maxLatticeInt; ++jj)
                {
                bool includePoint = inAnnulus(wavevectors[ii][jj],rmin,rmax);
                if (includePoint)
                    {
                    inSum +=1;
                    value += SK[ii][jj];
                    }
                };
            };
        if (inSum >0)
            {
            value = value/inSum;
            answer.push_back(make_double2(rmin+0.5*binWidth,value));
            };
        };

    SofK=answer;
    };
