#ifndef simpleEquationOfMotion_H
#define simpleEquationOfMotion_H

#include "simpleEquationOfMotion.cuh"
#include "gpuarray.h"
#include "updaterWithNoise.h"

/*! \file simpleEquationOfMotion.h */
//!A base class for implementing simple equations of motion
/*!
In cellGPU a "simple" equation of motion is one that can take a GPUArray of forces and return a set
of displacements. A derived class of this might be the self-propelled particle equations of motion,
or simple Brownian dynamics.
Derived classes must implement the integrateEquationsOfMotion function. Additionally, equations of
motion act on a cell configuration, and in general require that the configuration, C,  passed in to the
equation of motion provides access to the following:
C->getNumberOfDegreesOfFreedom() should return the number of degrees of freedom (up to a factor of
dimension)
C->computeForces() should calculate the negative gradient of the energy in whatever model T implements
C->getForces(f) is able to be called after T.computeForces(), and copies the forces to the variable f
C->moveDegreesOfFreedom(disp) moves the degrees of freedom according to the GPUArray of displacements
C->enforceTopology() takes care of any business the model that T implements needs after the
positions of the underlying degrees of freedom have been updated

*/
class simpleEquationOfMotion : public updaterWithNoise
    {
    public:
        //!base constructor sets default time step size
        simpleEquationOfMotion()
            {
            Period = 1;
            Phase = 0;
            deltaT = 0.01; GPUcompute =true;Timestep = 0;
            };
        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion(){};

        //!get the number of timesteps run
        int getTimestep(){return Timestep;};
        //!get the current simulation time
        Dscalar getTime(){return (Dscalar)Timestep * deltaT;};
        //!Set the simulation time stepsize
        virtual void setDeltaT(Dscalar dt){deltaT = dt;};
        //! performUpdate just maps to integrateEquationsOfMotion
        virtual void performUpdate(){integrateEquationsOfMotion();};

    protected:
        //! Count the number of integration timesteps
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;

        //!an internal GPUArray for holding displacements
        GPUArray<Dscalar2> displacements;
    };

typedef shared_ptr<simpleEquationOfMotion> EOMPtr;
typedef weak_ptr<simpleEquationOfMotion> WeakEOMPtr;

#endif
