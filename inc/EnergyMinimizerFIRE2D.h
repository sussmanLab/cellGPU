#ifndef ENERGYMINIMIZERFIRE_H
#define ENERGYMINIMIZERFIRE_H

#include "std_include.h"
#include "functions.h"
#include "gpuarray.h"
//include for explicit instantiation in the cpp file
#include "Simple2DActiveCell.h"


/*! \file EnergyMinimizerFIRE2D.h
This class uses the "FIRE" algorithm to perform an energy minimization. It requires that
the class, T, of the model provides access to the following functions:
T.getNumberOfDegreesOfFreedom() should return the number of degrees of freedom (up to a factor of
dimension)
T.computeForces() should calculate the negative gradient of the energy in whatever model T implements
T.getForces(f) is able to be called after T.computeForces(), and copies the forces to the variable f
T.moveDegreesOfFreedom(disp) moves the degrees of freedom according to the GPUArray of displacements
T.enforceTopology() takes care of any business the model that T implements needs after the
positions of the underlying degrees of freedom have been updated
*/
//!Implement energy minimization via the FIRE algorithm
class EnergyMinimizerFIRE
    {
    public:
        //!A no-initialization constructor for template instantiation
        EnergyMinimizerFIRE(){};

        //!The system that can compute forces, move degrees of freedom, etc.
        Simple2DActiveCell *State;

        //!The basic constructor that feeds in a target system to minimize
        EnergyMinimizerFIRE(Simple2DActiveCell &system);

        void setSystem(Simple2DActiveCell  &_sys){State = &_sys;};

        //!Set the maximum number of iterations before terminating (or set to -1 to ignore)
        void setMaximumIterations(int maxIt){maxIterations = maxIt;};

        //!Set the force cutoff
        void setForceCutoff(Dscalar fc){forceCutoff = fc;};

        //!set the initial value of deltaT
        void setDeltaT(Dscalar dt){deltaT = dt;};
        //!set the initial value of alpha and alphaStart
        void setAlphaStart(Dscalar as){alphaStart = as;alpha = as;};
        //!Set the maximum deltaT
        void setDeltaTMax(Dscalar tmax){deltaTMax = tmax;};
        //!Set the fraction by which delta increments
        void setDeltaTInc(Dscalar dti){deltaTInc = dti;};
        //!Set the fraction by which delta decrements
        void setDeltaTDec(Dscalar dtc){deltaTDec = dtc;};
        //!Set the fraction by which alpha decrements
        void setAlphaDec(Dscalar ad){alphaDec = ad;};
        //!Set the number of consecutive steps P must be non-negative before increasing delatT
        void setNMin(int nm){NMin = nm;};
        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        void setGPU(){GPUcompute = true;};
        //!Enforce CPU-only operation.
        void setCPU(){GPUcompute = false;};

        //!an interface to call either the CPU or GPU velocity Verlet algorithm
        void velocityVerlet();
        //!Perform a velocity Verlet step on the CPU
        void velocityVerletCPU();
        //!Perform a velocity Verlet step on the GPU
        void velocityVerletGPU();

        //!an interface to call either the CPU or GPU FIRE algorithm
        void fireStep();
        //!Perform a velocity Verlet step on the CPU
        void fireStepCPU();
        //!Perform a velocity Verlet step on the GPU
        void fireStepGPU();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();

        //!Test the parallel reduction routines by passing in a known vector
        void parallelReduce(GPUArray<Dscalar> &vec);

    protected:
        //!The number of iterations performed
        int iterations;
        //!The maximum number of iterations allowed
        int maxIterations;
        //!The cutoff value of the maximum force
        Dscalar forceMax;
        //!The cutoff value of the maximum force
        Dscalar forceCutoff;
        //!The number of points, or cells, or particles
        int N;
        //!The numer of consecutive time steps the power must be positive before increasing deltaT
        int NMin;
        //!The numer of consecutive time since the power has be negative
        int NSinceNegativePower;
        //!The internal time step size
        Dscalar deltaT;
        //!The maximum time step size
        Dscalar deltaTMax;
        //!The fraction by which deltaT can get bigger
        Dscalar deltaTInc;
        //!The fraction by which deltaT can get smaller
        Dscalar deltaTDec;
        //!The internal value of the "power"
        Dscalar Power;
        //!The alpha parameter of the minimization routine
        Dscalar alpha;
        //!The initial value of the alpha parameter
        Dscalar alphaStart;
        //!The fraction by which alpha can decrease
        Dscalar alphaDec;
        //!The GPUArray containing the force
        GPUArray<Dscalar2> force;
        //!The GPUArray containing the velocity
        GPUArray<Dscalar2> velocity;
        //!an array of displacements
        GPUArray<Dscalar2> displacement;

        //!Utility array for computing force.velocity
        GPUArray<Dscalar> forceDotVelocity;
        //!Utility array for computing force.force
        GPUArray<Dscalar> forceDotForce;
        //!Utility array for computing velocity.velocity
        GPUArray<Dscalar> velocityDotVelocity;

        //!Utility array for simple reductions
        GPUArray<Dscalar> sumReductionIntermediate;
        //!Utility array for simple reductions
        GPUArray<Dscalar> sumReductions;

        //!Should calculations be done on the GPU?
        bool GPUcompute;

    };

#endif
