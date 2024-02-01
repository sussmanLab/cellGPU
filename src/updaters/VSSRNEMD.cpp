#include "VSSRNEMD.h"
/*! \file VSSRNEMD.cpp */

/*!
Initialize the updater by setting the period of updates, the kinetic energy and x momentum
transferred between slabs in each iteration of the updater, and the boundaries of the two slabs
involved in swaps.
*/
VSSRNEMD::VSSRNEMD(int period, double dt, int slabs, double boxHeight, double pxFlux, double KEFlux)
    {
    setPeriod(period);
    KETransfer = 2*KEFlux*boxHeight*period*dt;
    pxTransfer = 2*pxFlux*boxHeight*period*dt;
    printf("x momentum transferred in each cycle: %f\n",pxTransfer);
    printf("Kinetic energy transferred in each cycle: %f\n",KETransfer);

    accumulatedDeltaPx = 0.0;
    accumulatedDeltaKE = 0.0;
    updateSlabInfo(slabs,boxHeight);
    };

/*!
update the geometry of the slabs by specifying the number of slabs and the box height
The two swapping slabs will be at the bottom of the simulation box and in the middle of it
*/
void VSSRNEMD::updateSlabInfo(int slabs, double boxHeight)
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
Perform velocity scaling and shearing updates on particles in the central and lowest 
slices of the simulation box
*/
void VSSRNEMD::performUpdate()
    {
    ArrayHandle<double2> h_p(model->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<double> h_m(model->returnMasses(),access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(model->returnVelocities());

    //Mass, x and y momenta, x and y kKE of cold slab
    double mc=0, pcx=0, pcy=0, KEcx=0, KEcy=0; 
    //and hot slab
    double mh=0, phx=0, phy=0, KEhx=0, KEhy=0; 

    vector<int> hotSlice, coldSlice; //Store ids of particles in hot and cold slices

    int N = model->getNumberOfDegreesOfFreedom();
    for (int ii = 0; ii < N; ++ii)
        {
        double y = h_p.data[ii].y;

        // Find particles in cold slab, calculate their total momentum, KE and mass
        if(y>slabBoundaries.x && y < slabBoundaries.y) 
            {
            coldSlice.push_back(ii);
            mc += h_m.data[ii];
            pcx += h_m.data[ii]*h_v.data[ii].x;
            pcy += h_m.data[ii]*h_v.data[ii].y;
            KEcx += 0.5*h_m.data[ii]*h_v.data[ii].x*h_v.data[ii].x;
            KEcy += 0.5*h_m.data[ii]*h_v.data[ii].y*h_v.data[ii].y;
            }

        // Find particles in hot slab, calculate their total momentum, KE and mass    
        if(y>slabBoundaries.z && y < slabBoundaries.w) 
            {
            hotSlice.push_back(ii);
            mh += h_m.data[ii];
            phx += h_m.data[ii]*h_v.data[ii].x;
            phy += h_m.data[ii]*h_v.data[ii].y;
            KEhx += 0.5*h_m.data[ii]*h_v.data[ii].x*h_v.data[ii].x;
            KEhy += 0.5*h_m.data[ii]*h_v.data[ii].y*h_v.data[ii].y;
            };
        }

    // Total KE in the two slabs
    double KEc = KEcx + KEcy;
    double KEh = KEhx + KEhy;

    if(mc<=0 || mh<=0)
        {
        printf("RNEMD VSS failed: No particles in one of the slabs");
        }

    if(mc > 0 && mh > 0) // Check both slabs contain particles
        {
        // Average velocity in each slab
        double vcx = pcx/mc;
        double vcy = pcy/mc;
        double vhx = phx/mh;
        double vhy = phy/mh;

        // Shearing term in each slab
        double scx = vcx - pxTransfer/mc;
        double scy = vcy;
        double shx = vhx + pxTransfer/mh;
        double shy = vhy; 

        // Solve for c and h coefficients
        double cNumerator = KEc - KETransfer - 0.5*mc*(scx*scx+scy*scy);
        double cDenominator = KEc - 0.5*mc*(vcx*vcx+vcy*vcy);

        double hNumerator = KEh + KETransfer - 0.5*mh*(shx*shx+shy*shy);
        double hDenominator = KEh - 0.5*mh*(vhx*vhx+vhy*vhy);

        if(cDenominator <=0 || hDenominator <= 0)
            {
            printf("VSS RNEMD failed: c denominator: %f, h denominator: %f",cDenominator,hDenominator);
            }

        if(cNumerator <=0 || hNumerator <= 0)
            {
            printf("c numerator: %f, h numerator: %f",cNumerator,hNumerator);
            }    

        if(cDenominator > 0 && hDenominator > 0)
            {
            double c = sqrt(cNumerator/cDenominator);
            double h = sqrt(hNumerator/hDenominator);

            // Restrict scaling coefficients to be close to 1
            if(c > 0.9 && c < 1.1 && h > 0.9 && h < 1.1) 
                {
                //Perform the shearing and rescaling
                for(int ii : coldSlice)
                    {
                    h_v.data[ii].x = c*(h_v.data[ii].x-vcx) + scx;
                    h_v.data[ii].y = c*(h_v.data[ii].y-vcy) + scy;
                    }
                

                for(int ii : hotSlice)
                    {
                    h_v.data[ii].x = h*(h_v.data[ii].x-vhx) + shx;
                    h_v.data[ii].y = h*(h_v.data[ii].y-vhy) + shy;
                    }
                
                accumulatedDeltaKE += KETransfer;
                accumulatedDeltaPx += pxTransfer;

                }

            else
                {
                printf("VSS RNEMD failed: c: %f, h: %f\n",c,h);
                }    

            }

        }


    };


/*!
This function takes the particles, sorts them into slabs, and gets the average x-direction
velocity in each slab.
*/
void VSSRNEMD::getVelocityProfile(vector<double> &Vx)
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
        
/*!Returns the total x-momentum transferred between the slabs since the last 
time this function was called
*/
double VSSRNEMD::getxMomentumTransferred()
    {
    double answer = accumulatedDeltaPx;
    accumulatedDeltaPx = 0.0;
    return answer;
    };

/*!Returns the total kinetic energy transferred between the slabs since the last 
time this function was called
*/
double VSSRNEMD::getKineticEnergyTransferred()
    {
    double answer = accumulatedDeltaKE;
    accumulatedDeltaKE = 0.0;
    return answer;
    };       

