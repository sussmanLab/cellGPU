#ifndef ENERGYMINIMIZERFIRE_H
#define ENERGYMINIMIZERFIRE_H

#include "functions.h"
#include "gpuarray.h"
#include "Simple2DCell.h"
#include "simpleEquationOfMotion.h"


/*! \file EnergyMinimizerFIRE2D.h */
//!Implement energy minimization via the FIRE algorithm
/*!
This class uses the "FIRE" algorithm to perform an energy minimization.
The class is in the same framework as the simpleEquationOfMotion class, so it is called by calling
performTimestep on a Simulation object that has been given the FIRE minimizer and the configuration
to minimize. Each timestep, though, is a complete minimization (i.e. will run for the maximum number
of iterations, or until a target tolerance has been acheived, or whatever stopping condition the user
sets.
*/
class EnergyMinimizerFIRE : public simpleEquationOfMotion
    {
    public:
        //!The basic constructor
        EnergyMinimizerFIRE(){initializeParameters();};
        //!The basic constructor that feeds in a target system to minimize
        EnergyMinimizerFIRE(shared_ptr<Simple2DModel> system);
        //!Sets a bunch of default parameters that do not depend on the number of degrees of freedom
        void initializeParameters();
        //!Set a bunch of default initialization parameters (if the State is available to determine the size of vectors)
        void initializeFromModel();

        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){State = _model;};

        //!Set the maximum number of iterations before terminating (or set to -1 to ignore)
        void setMaximumIterations(int maxIt){maxIterations = maxIt;};
        //!Set the force cutoff
        void setForceCutoff(Dscalar fc){forceCutoff = fc;};
        //!set the initial value of deltaT
        void setDeltaT(Dscalar dt){deltaT = dt;deltaTMin=dt*.01;};
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
        //!The "intergate equatios of motion just calls minimize
        virtual void integrateEquationsOfMotion(){minimize();};

        //!Test the parallel reduction routines by passing in a known vector
        void parallelReduce(GPUArray<Dscalar> &vec);

        //!Return the maximum force
        Dscalar getMaxForce(){return forceMax;};

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
        //!The minimum time step size
        Dscalar deltaTMin;
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
    };
#endif
