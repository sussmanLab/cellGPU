#ifndef simpleEquationOfMotion_H
#define simpleEquationOfMotion_H

#include "std_include.h"
#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
#include "gpuarray.h"
#include "gpubox.h"
#include "curand.h"
#include "curand_kernel.h"

/*! \file simpleEquationOfMotion.h
In cellGPU a "simple" equation of motion is one that can take a GPUArray of forces and return a set
of displacements. A derived class of this might be the self-propelled particle equations of motion,
or simple Brownian dynamics.
Derived classes must implement the integrateEquationsOfMotion function
*/
//!A base class for implementing simple equations of motion
class simpleEquationOfMotion
    {
    public:
        //!base constructor sets default time step size
        Simple2DActiveCell(){deltaT = 0.01; GPUcompute =true;};

        //!the fundamental function that models will call to advance the simulation
        virtual void integrateEquationsOfMotion(GPUArray<Dscalar2> &forces, GPUArray<Dscalar2> &displacements);

        //!get the number of timesteps run
        int getTimestep(){return Timestep;};
        //!get the current simulation time
        int getTime(){return (Dscalar)Timestep * deltaT;};
        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};
        //!Get the number of degrees of freedom of the equation of motion
        int getNdof(){return Ndof;};
        //!Set the number of degrees of freedom of the equation of motion
        void setNdof(int _n){Ndof = _n;};

        //NEED TO PROVIDE THE RIGHT SPATIAL SORTING CAPABILITIES (IF, EG, MOTILITY IS IN THE EOMs)

        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU(){GPUcompute = true;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};

    protected:
        //!The number of degrees of freedom the equations of motion need to know about
        int Ndof;
        //! Count the number of integration timesteps
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;
    };

#endif
