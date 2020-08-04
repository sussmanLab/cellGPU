#ifndef SIMPLEMODEL_H
#define SIMPLEMODEL_H

#include "std_include.h"
#include "gpuarray.h"

/*! \file Simple2DModel.h
 * \brief defines an interface for models that compute forces
 */

//! A base interfacing class that defines common operations
/*!
This provides an interface, guaranteeing that SimpleModel S will provide access to
S.getNumberOfDegreesOfFreedom();
S.computeForces();
S.getMaxForce();
S.getDynMatEntries();
S.getForces();
S.moveDegreesOfFreedom();
S.enforceTopology();
S.spatialSorting();
S.returnVelocities();
S.returnMasses();
S.returnOtherData(); //this last will be a flat GPUArray of doubles...
*/
class Simple2DModel
    {
    public:
        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU() = 0;
        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU() = 0;
        //!get the number of degrees of freedom, defaulting to the number of cells
        virtual int getNumberOfDegreesOfFreedom() = 0;
        //!do everything necessary to compute forces in the current model
        virtual void computeForces() = 0;
        //!Do whatever is necessary to get lists of dynamical matrix elements
        virtual void getDynMatEntries(vector<int2> &rcs, vector<double> &vals,double unstress = 1.0, double stress = 1.0){};
        //!do everything necessary to perform a Hilbert sort
        virtual void spatialSorting(){};
        //!do everything necessary to enforce the topology of the system
        virtual void enforceTopology(){};
        //!copy the models current set of forces to the variable
        virtual void getForces(GPUArray<double2> &forces) = 0;
        //!Return the maximum force
        virtual double getMaxForce(){return 0.;};
        //!return a reference to the GPUArray of positions
        virtual GPUArray<double2> & returnPositions() = 0;
        //!return a reference to the GPUArray of the masses
        virtual GPUArray<double> & returnMasses() = 0;
        //!return a reference to the GPUArray of other data (definable as needed in child classes)
        virtual GPUArray<double> & returnOtherData() = 0;
        //!return a reference to the GPUArray of the current velocities
        virtual GPUArray<double2> & returnVelocities() = 0;
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<double2> & returnForces() = 0;
        //!move the degrees of freedom
        virtual void moveDegreesOfFreedom(GPUArray<double2> &displacements,double scale = 1.) = 0;
        //!reporting function (remove later...)
        virtual double reportq() = 0;
        //!reporting function (remove later...)
        virtual void reportMeanCellForce(bool a) = 0;
        //!a time variable for keeping track of the simulation variable (for databases)
        double currentTime;
        //!set the time
        virtual void setTime(double time){currentTime = time;};

        //!Allow openMP threads
        int ompThreadNum = 1;
        //set number of threads
        virtual void setOmpThreads(int _number){ompThreadNum = _number;};
    };
#endif
