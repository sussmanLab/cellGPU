#include "MullerPlatheShear.h"
/*! \file MullerPlatheShear.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
MullerPlatheShear::MullerPlatheShear(int period, int slabs,double boxHeight)
    {
    setPeriod(period);
    deltaP = 0;
    accumulatedDeltaP = 0.0;
    updateSlabInfo(slabs,boxHeight);
    };

/*!
update the geometry of the slabs by specifying the number of slabs and the box height
The two swapping slabs will be at the bottom of the simulation box and in the middle of it
*/
void MullerPlatheShear::updateSlabInfo(int slabs, double boxHeight)
    {
    Nslabs = floor(slabs/2)*2;
    boxY = boxHeight;
    slabThickness = (double)boxY/Nslabs;
    slabBoundaries.x = 0.0;
    slabBoundaries.y = slabThickness;
    slabBoundaries.z = Nslabs/2.0*slabThickness;
    slabBoundaries.w = (1+Nslabs/2.0)*slabThickness;
    };

/*
Step through every particle in the simulation and try to identify particles moving "against" the
slabs' desired direction. Swap the x-momenta of the two worst offenders
*/
void MullerPlatheShear::performUpdate()
    {
    //first, try to identify the right-fastest and left-fastest particles in the two slabs
    int p1idx = -1;
    int p2idx = -1;
    double maxPositiveV = -1000000.;
    double maxNegativeV =  1000000.;
    ArrayHandle<double2> h_p(model->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<double> h_m(model->returnMasses(),access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(model->returnVelocities());
    int N = model->getNumberOfDegreesOfFreedom();
    for (int ii = 0; ii < N; ++ii)
        {
        double y = h_p.data[ii].y;
        if(y>slabBoundaries.x && y < slabBoundaries.y)
            {
            if (h_v.data[ii].x > maxPositiveV)
                {
                p1idx = ii;
                maxPositiveV = h_v.data[ii].x;
                };
            };
        if(y>slabBoundaries.z && y < slabBoundaries.w)
            {
            if (h_v.data[ii].x < maxNegativeV)
                {
                p2idx = ii;
                maxNegativeV = h_v.data[ii].x;
                };
            };
        };

    //if swap candidates were found, swap the velocities
    if(p1idx >=0 && p2idx >=0 && maxNegativeV < 0 && maxPositiveV > 0)
        {
        double v1 = h_v.data[p1idx].x;
        double m1 = h_m.data[p1idx];
        double v2 = h_v.data[p2idx].x;
        double m2 = h_m.data[p2idx];
        h_v.data[p2idx].x = v1;
        h_v.data[p1idx].x = v2;
        deltaP = m1*v1-m2*v2;
        accumulatedDeltaP += deltaP;
        };
    };

/*!
This function takes the particles, sorts them into slabs, and gets the average x-direction
velocity in each slab.
*/
void MullerPlatheShear::getVelocityProfile(vector<double> &Vx)
    {
    vector<int> particlesPerSlab(Nslabs,0);
    Vx.resize(Nslabs);
    for (int ii = 0; ii < Nslabs; ++ii)
        Vx[ii] = 0.0;

    ArrayHandle<double2> h_p(model->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(model->returnVelocities());
    int N = model->getNumberOfDegreesOfFreedom();
    for (int ii = 0; ii < N; ++ii)
        {
        double y = h_p.data[ii].y;
        int slabIdx = floor(y/slabThickness);
        if(slabIdx >=0 && slabIdx < Nslabs)
            {
            Vx[slabIdx] += h_v.data[ii].x;
            particlesPerSlab[slabIdx] += 1;
            };
        };
    for (int ii = 0; ii < Nslabs; ++ii)
        if(particlesPerSlab[ii]>0)
            Vx[ii] = Vx[ii] / particlesPerSlab[ii];
    };
