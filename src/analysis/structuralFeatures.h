#ifndef structuralFeatures_H
#define structuralFeatures_H

#include "std_include.h"
#include "functions.h"
#include "periodicBoundaries.h"
#include "indexer.h"

/*! \file structuralFeatures.h */

//! A class that calculates various structural features of 2D point patterns
class structuralFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        structuralFeatures(PeriodicBoxPtr _bx){Box = _bx;};
        structuralFeatures(){};

        void setBox(PeriodicBoxPtr _bx){Box = _bx;};

        //!Compute the (isotropic) radial distribution function of the point pattern
        void computeRadialDistributionFunction(vector<double2> &points,vector<double2> &GofR, double binWidth = 0.1);

        //!Compute the (isotropic) structure factor out to some maximum value of k
        void computeStructureFactor(vector<double2> &points, vector<double2> &SofK, double intKMax = 1.0,double dk = 0.5);
        
        //!Compute the angular bond order parameter. default to hexatic. result.x is real part, result.y is imaginary
        double2 computeBondOrderParameter(GPUArray<double2> &points, GPUArray<int> &neighbors, GPUArray<int> &neighborNum, Index2D n_idx, int n = 6);
    protected:
        //!the box defining the periodic domain
        PeriodicBoxPtr Box;
    };
#endif
