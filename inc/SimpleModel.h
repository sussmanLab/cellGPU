#ifndef SIMPLEMODEL_H
#define SIMPLEMODEL_H

#include "std_include.h"
#include "gpuarray.h"

/*! \file SimpleModel.h
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
class SimpleModel
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

        //!do everything necessary to compute forces in the current model
        virtual void getForces(GPUArray<Dscalar2> &forces) = 0;

        //!move the degrees of freedom
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements) = 0;
    };

#endif
