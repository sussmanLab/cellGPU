#ifndef ENERGYMINIMIZERFIRE_H
#define ENERGYMINIMIZERFIRE_H

#include "std_include.h"
#include "gpuarray.h"

/*! \file EnergyMinimizerFire.h
This templated class uses the "FIRE" algorithm to perform an energy minimization. It requires that
the templated class, T, provides access to the following functions:
T.getNumberOfDegreesOfFreedom() should return the number of degrees of freedom (up to a factor of
dimension)
T.computeForces() should calculate the negative gradient of the energy in whatever model T implements
T.getForces(f) is able to be called after T.computeForces(), and copies the forces to the variable f
T.moveDegreesOfFreedom(disp) moves the degrees of freedom according to the GPUArray of displacements
T.enforceTopology() takes care of any business the model that T implements needs after the
positions of the underlying degrees of freedom have been updated
*/
//!Implement energy minimization via the FIRE algorithm
template <class T>
class EnergyMinimizerFIRE
    {
    public:
        //!The basic constructor that feeds in a target system to minimize
        EnergyMinimizerFIRE(T &system);

        //!Set the maximum number of iterations before terminating (or set to -1 to ignore)
        setMaximumIterations(int maxIt){maxIterations = maxIt;};

        //!Set the force cutoff
        setForceCutoff(Dscalar fc){forceMax = fc;};

        //!set the initial value of deltaT
        setDeltaT(Dscalar dt){deltaT = dt;};
        //!set the initial value of alpha and alphaStart
        setAlphaStart(Dscalar as){alphaStart = as;alpha = as;};
        //!Set the maximum deltaT
        setDeltaTMax(Dscalar tmax){deltaTMax = tmax;};
        //!Set the fraction by which delta increments
        setDeltaTMax(Dscalar dti){deltaTInc = dti;};
        //!Set the fraction by which delta decrements
        setDeltaTMax(Dscalar dtd){deltaTDec = dtc;};
        //!Set the fraction by which alpha decrements
        setDeltaTMax(Dscalar ad){alphaDec = ad;};
        //!Set the number of consecutive steps P must be non-negative before increasing delatT
        setNMin(int nm){NMin = nm;};

    private:
        //!The system that can compute forces, move degrees of freedom, etc.
        T State;
        //!The maximum number of iterations allowed
        int maxIterations;
        //!The cutoff value of the maximum force
        Dscalar forceMax;
        //!The number of points, or cells, or particles
        int N;
        //!The numer of consecutive time steps the power must be positive before increasing deltaT
        int NMin;
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

    };

template <class T>
EnergyMinimizaerFIRE<T>::EnergyMinimizerFIRE(T &system)
            :State(&system)
    {
    N = State.getNumberOfDegreesOfFreedom();
    force.resize(N);
    velocity.resize(N);
    ArrayHandle<Dscalar2> h_f(force);
    ArrayHandle<Dscalar2> h_v(velocity);
    Dscalar2 zero; zero.x = 0.0; zero.y = 0.0;
    for(int i = 0; i <N; ++i)
        {
        h_f.data[i]=zero;
        h_v.data[i]=zero;
        };
    Power = 0;
    setMaximumIterations(1000);
    setForceCutoff(1e-7);
    setAlphaStart(0.99);
    setDeltaT(0.01);

    };

#endif

