#ifndef SIMPLEMODEL_H
#define SIMPLEMODEL_H

#include "std_include.h"
#include "gpuarray.h"
//#include "simpleEquationOfMotion.h"
class Simulation;

/*! \file Simple2DModel.h
 * \brief defines an interface for models that compute forces
 */

//! A base interfacing class that defines common operations
/*!
This provides an interface, guaranteeing that SimpleModel S will provide access to
S.getNumberOfDegreesOfFreedom();
S.computeForces();
S.getForces();
S.moveDegreesOfFreedom();
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

        //!copy the models current set of forces to the variable
        virtual void getForces(GPUArray<Dscalar2> &forces) = 0;
        
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<Dscalar2> & returnForces() = 0;

        //!move the degrees of freedom
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements) = 0;
        //!reporting function (remove later...)
        virtual Dscalar reportq() = 0;
        virtual void reportMeanCellForce(bool a) = 0;
        virtual void setSortPeriod(int a) = 0;
        virtual void performTimestep() = 0;

        //! pointer to a Simulation
        shared_ptr<Simulation> simulation;
        //!An EOMPtr (remove later...)
        //EOMPtr equationOfMotion; 

//        void setEquationOfMotion(EOMPtr &_eom){equationOfMotion = _eom;};
    };


#endif
