#ifndef structuralFeatures_H
#define structuralFeatures_H

#include "std_include.h"
#include "functions.h"
#include "gpubox.h"

/*! \file structuralFeatures.h */

//! A class that calculates various structural features of 2D point patterns
class structuralFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        structuralFeatures(BoxPtr _bx){Box = _bx;};

        //!Compute the (isotropic) radial distribution function of the point pattern
        void computeRadialDistributionFunction(vector<Dscalar2> &points,vector<Dscalar2> &GofR, Dscalar binWidth = 0.1);

        //!Compute the (isotropic) structure factor out to some maximum value of k
        void computeStructureFactor(vector<Dscalar2> &points, vector<Dscalar2> &SofK, Dscalar intKMax = 1.0,Dscalar dk = 0.5);
    protected:
        //!the box defining the periodic domain
        BoxPtr Box;
    };
#endif
