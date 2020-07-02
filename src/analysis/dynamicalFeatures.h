#ifndef dynamicalFeatures_H
#define dynamicalFeatures_H

#include "std_include.h"
#include "functions.h"
#include "gpubox.h"

/*! \file dynamicalFeatures.h */

//! A class that calculates various dynamical features for 2D systems
class dynamicalFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        dynamicalFeatures(GPUArray<Dscalar2> &initialPos, BoxPtr _bx, Dscalar fractionAnalyzed = 1.0);

        //!Compute the mean squared displacement of the passed vector from the initial positions
        Dscalar computeMSD(GPUArray<Dscalar2> &currentPos);

        //!compute the overlap function
        Dscalar computeOverlapFunction(GPUArray<Dscalar2> &currentPos, Dscalar cutoff = 0.5);
    protected:
        //!the box defining the periodic domain
        BoxPtr Box;
        //!the initial positions
        vector<Dscalar2> iPos;
        //!the number of Dscalar2's
        int N;
    };
#endif
