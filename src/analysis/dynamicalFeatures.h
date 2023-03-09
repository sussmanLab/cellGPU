#ifndef dynamicalFeatures_H
#define dynamicalFeatures_H

#include "std_include.h"
#include "functions.h"
#include "periodicBoundaries.h"

/*! \file dynamicalFeatures.h */

//! A class that calculates various dynamical features for 2D systems
class dynamicalFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        dynamicalFeatures(GPUArray<double2> &initialPos, PeriodicBoxPtr _bx, double fractionAnalyzed = 1.0);

        //!Compute the mean squared displacement of the passed vector from the initial positions
        double computeMSD(GPUArray<double2> &currentPos);

        //!compute the overlap function
        double computeOverlapFunction(GPUArray<double2> &currentPos, double cutoff = 0.5);
    protected:
        //!the box defining the periodic domain
        PeriodicBoxPtr Box;
        //!the initial positions
        vector<double2> iPos;
        //!the number of double2's
        int N;
    };
#endif
