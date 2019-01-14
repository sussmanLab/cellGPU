#ifndef SIMPLE2DACTIVITY_H
#define SIMPLE2DACTIVITY_H

#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
#include "indexer.h"
#include "curand.h"
#include "curand_kernel.h"

/*! \file Simple2DActiveCell.h */
//!Data structures and functions for simple active-brownian-particle-like motion
/*!
A class defining the simplest aspects of a 2D system in which particles have a constant velocity
along a director which rotates with gaussian noise
*/
class Simple2DActiveCell : public Simple2DCell
    {
    //public functions first
    public:
        //!A simple constructor
        Simple2DActiveCell();

        //! initialize class' data structures and set default values
        void initializeSimple2DActiveCell(int n);

        //!Set uniform motility
        void setv0Dr(Dscalar v0new,Dscalar drnew);

        //!Set non-uniform cell motilites
        void setCellMotility(vector<Dscalar> &v0s,vector<Dscalar> &drs);

        //!Set random cell directors (for active cell models)
        void setCellDirectorsRandomly();

        //!get the number of degrees of freedom, defaulting to the number of cells
        virtual int getNumberOfDegreesOfFreedom(){return Ncells;};

        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(const vector<int> &parameters, const vector<Dscalar> &dParams = {});

        //!Kill the indexed cell
        virtual void cellDeath(int cellIndex);

        //!measure the viscek order parameter N^-1 \sum \frac{v_i}{|v_i}
        Dscalar vicsekOrderParameter(Dscalar2 &vParallel, Dscalar2 &vPerpendicular)
            {
            ArrayHandle<Dscalar2> vel(returnVelocities());
            Dscalar2 globalV = make_Dscalar2(0.0,0.0);
            for(int ii = 0; ii < getNumberOfDegreesOfFreedom(); ++ii)
                {
                Dscalar2 v = (1.0/norm(vel.data[ii]))*vel.data[ii];
                vParallel = vParallel+vel.data[ii];
                globalV = globalV+v;
                };

            vParallel = (1.0/norm(vParallel))*vParallel;
            vPerpendicular.x= -vParallel.y;
            vPerpendicular.y=  vParallel.x;
            return norm(globalV)/getNumberOfDegreesOfFreedom();
            };

        //!measure the viscek order parameter N^-1 \sum \frac{v_i}{|v_i} from the director only
        Dscalar vicsekOrderParameterDirector(Dscalar2 &vParallel, Dscalar2 &vPerpendicular)
            {
            ArrayHandle<Dscalar> cd(cellDirectors);
            Dscalar2 globalV = make_Dscalar2(0.0,0.0);
            Dscalar thetaAve = 0.0;
            for(int ii = 0; ii < getNumberOfDegreesOfFreedom(); ++ii)
                {
                Dscalar theta=cd.data[ii];
                if(theta < -PI)
                    theta += 2*PI;
                if(theta > PI)
                    theta -= 2*PI;
                thetaAve += theta;
                Dscalar2 v;
                v.x = cos(theta);
                v.y = sin(theta);
                globalV = globalV+v;
                };

            globalV.x /= getNumberOfDegreesOfFreedom();
            globalV.y /= getNumberOfDegreesOfFreedom();
            thetaAve /= getNumberOfDegreesOfFreedom();

            vParallel.x= cos(thetaAve);
            vParallel.y= sin(thetaAve);
            
            vPerpendicular.x= -vParallel.y;
            vPerpendicular.y=  vParallel.x;
            
            return sqrt(globalV.x*globalV.x + globalV.y*globalV.y);
            };


    //protected functions
    protected:
        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        void spatiallySortVerticesAndCellActivity();

        //!call the Simple2DCell spatial cell sorter, and re-index arrays of cell activity
        void spatiallySortCellsAndCellActivity();

    //public member variables
    public:
        //!An array of angles (relative to the x-axis) that the cell directors point
        GPUArray<Dscalar> cellDirectors;
        //!An array of forces acting on the cell directors
        GPUArray<Dscalar> cellDirectorForces;

        //!velocity of cells in mono-motile systems
        Dscalar v0;
        //!rotational diffusion of cell directors in mono-motile systems
        Dscalar Dr;
        //!The motility parameters (v0 and Dr) for each cell
        GPUArray<Dscalar2> Motility;
    };
#endif
