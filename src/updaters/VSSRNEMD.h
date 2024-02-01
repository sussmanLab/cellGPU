#ifndef VSSRNEMD_H
#define VSSRNEMD_H

#include "std_include.h"
#include "updater.h"

/*! \file VSSRNEMD.h */
//!An updater that creates a velocity profile by velocity scaling and shearing particles in chosen regions
/*!
This updater implements the Velocity Scaling and Shearing Reverse Non-Equilibrium Molecular Dynamics 
(VSS-RNEMD) method from S. Kuang and J. D. Gezelter, Mol, Phys., 110, 691 (2012).
The y-direction is partitioned into slabs with a "hot" center slab and "cold" slab at y=0.
Velocities in these slabs are sheared and rescaled to impose a thermal gradient, momentum gradient or both
along the y-direction of the system.
*/
class VSSRNEMD : public updater
    {
    public:
        //!The basic call should specify the updater period (in units of dt), time step, the number of slabs, 
        //the y-height of the box, the momentum flux and the kinetic energy flux
        VSSRNEMD(int period, double dt, int slabs, double boxHeight, double pxFlux, double KEFlux);

        //!update the slab thicknesses
        void updateSlabInfo(int slabs,double boxHeight);

        //!perform the desired swapping
        virtual void performUpdate();

        //!The kinetic energy transferred each time the updater is called
        double KETransfer;
        //!The x-momentum transferred each time the updater is called
        double pxTransfer;
        //!The x-momentum transferred since the last time the updater was called
        double accumulatedDeltaPx;
        //!The kinetic energy transferred since the last time the updater was called
        double accumulatedDeltaKE;

        //!Returns the total x-momentum transferred between the slabs since the last time this function was called
        double getxMomentumTransferred();
  
        //!Returns the total kinetic energy transferred between the slabs since the last time this function was called
        double getKineticEnergyTransferred();

        //Averages the x-velocity in each slab and puts it in the vector
        void getVelocityProfile(vector<double> &Vx);
        //!The number of slabs in the y-direction... this implicitly controls the thickness of the slabs
        int Nslabs;

    protected:
        //!The size of the box in the y-directions
        double boxY;
        //!The thickness of the slabs
        double slabThickness;
        //!The lower and upper boundaries of the two slabs betw een which we will be swapping momenta
        double4 slabBoundaries;

    };
#endif
