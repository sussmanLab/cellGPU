#ifndef MullerPlatheShear_H
#define MullerPlatheShear_H

#include "std_include.h"
#include "updater.h"

/*! \file MullerPlateShear.h */
//!An updater that creates a velocity profile by swapping particle momenta
/*!
This updater implements the idea of M{\"u}ller-Plathe for computing viscosities by imposing
a momentum transfer scheme (and the measuring the resulting velocity profile). The idea is taken from:
F. M{\"u}ller-Plathe; Phys. Rev. E 59, 4894 (1999)
Briefly, the y-direction is partitioned into slabs, and the particle with the largest +x momentum in
one is swapped with the particle with the largest -x momentum in the other. Currently this updater
requires that the box by an orthogonal one
*/
class MullerPlatheShear : public updater
    {
    public:
        //!The basic call should specify the period, the number of slabs, and the y-height of the box
        MullerPlatheShear(int period, int slabs,Dscalar boxHeight);

        //!update the slab thicknesses
        void updateSlabInfo(int slabs,Dscalar boxHeight);

        //!perform the desired swapping
        virtual void performUpdate();

        //!The momentum transferred the last time the updater was called
        Dscalar deltaP;
        //!The momentum transferred since the last time the updater was querried
        Dscalar accumulatedDeltaP;

        //!Returns the total momentum transferred between the slabs since the last time this function was called
        Dscalar getMomentumTransferred()
            {
            Dscalar answer = accumulatedDeltaP;
            accumulatedDeltaP = 0.0;
            return answer;
            };
        //Averages the x-velocity in each slab and puts it in the vector
        void getVelocityProfile(vector<Dscalar> &Vx);
        //!The number of slabs in the y-direction... this implicitly controls the thickness of the slabs
        int Nslabs;

    protected:
        //!The size of the box in the y-directions
        Dscalar boxY;
        //!The thickness of the slabs
        Dscalar slabThickness;
        //!The lower and upper boundaries of the two slabs between which we will be swapping momenta
        Dscalar4 slabBoundaries;

    };
#endif
