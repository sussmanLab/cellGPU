#ifndef structuralFeatures_H
#define structuralFeature_H

#include "std_include.h"
#include "gpubox.h"

/*! \file structural.h */

//! A class that calculates various structural features of 2D point patterns
class structuralFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        structuralFeatures(BoxPtr _bx){Box = _box;};

        //!Compute the (isotropic) radial distribution function
        void computeRadialDistributionFunction(vector<Dscalar2> &points, Dscalar binWidth = 0.1,vector<Dscalar2> &GofR);
    protected:
        //!the box defining the periodic domain
        BoxPtr Box;
    };
#endif
