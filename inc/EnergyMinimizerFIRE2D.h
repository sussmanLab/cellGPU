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
        void velocityVerlet()
            {
            if (GPUcompute) velocityVerletGPU(); else velocityVerletCPU();
            };
        //!Perform a velocity Verlet step on the CPU
        void velocityVerletCPU();
        //!Perform a velocity Verlet step on the GPU
        void velocityVerletGPU();

        //!an interface to call either the CPU or GPU FIRE algorithm
        void fireStep()
            {
            if (GPUcompute) fireStepGPU(); else fireStepCPU();
            };
        //!Perform a velocity Verlet step on the CPU
        void fireStepCPU();
        //!Perform a velocity Verlet step on the GPU
        void fireStepGPU();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();

    protected:
        //!The system that can compute forces, move degrees of freedom, etc.
        T *State;
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

        //!Should calculations be done on the GPU?
        bool GPUcompute;

    };

/*!
Initialize the minimizer with a reference to a target system, set a bunch of default parameters.
Of note, the current default is CPU operation
*/
template <class T>
EnergyMinimizerFIRE<T>::EnergyMinimizerFIRE(T &system)
            :State(&system)
    {
    N = State->getNumberOfDegreesOfFreedom();
    force.resize(N);
    velocity.resize(N);
    displacement.resize(N);
    ArrayHandle<Dscalar2> h_f(force);
    ArrayHandle<Dscalar2> h_v(velocity);
    Dscalar2 zero; zero.x = 0.0; zero.y = 0.0;
    for(int i = 0; i <N; ++i)
        {
        h_f.data[i]=zero;
        h_v.data[i]=zero;
        };
    iterations = 0;
    Power = 0;
    NSinceNegativePower = 0;
    forceMax = 100.;
    setMaximumIterations(1000);
    setForceCutoff(1e-7);
    setAlphaStart(0.99);
    setDeltaT(0.01);
    setDeltaTMax(.1);
    setDeltaTInc(1.05);
    setDeltaTDec(0.95);
    setAlphaDec(.9);
    setNMin(5);
    setCPU();
    };

/*!
 * Perform a velocity verlet integration step on the GPU
 */
template <class T>
void EnergyMinimizerFIRE<T>::velocityVerletGPU()
    {
    };


/*!
 * Perform a velocity verlet integration step on the CPU
 */
template <class T>
void EnergyMinimizerFIRE<T>::velocityVerletCPU()
    {
    if(true) // scope for array handles
        {
        ArrayHandle<Dscalar2> h_f(force);
        ArrayHandle<Dscalar2> h_v(velocity);
        ArrayHandle<Dscalar2> h_d(displacement);
        for (int i = 0; i < N; ++i)
            {
            //update displacement
            h_d.data[i].x = deltaT*h_v.data[i].x+0.5*deltaT*deltaT*h_f.data[i].x;
            h_d.data[i].y = deltaT*h_v.data[i].y+0.5*deltaT*deltaT*h_f.data[i].y;
            //do first half of velocity update
            h_v.data[i].x += 0.5*deltaT*h_f.data[i].x;
            h_v.data[i].y += 0.5*deltaT*h_f.data[i].y;
            };
        };
    //move particles, then update the forces
    State->moveDegreesOfFreedom(displacement);
    State->enforceTopology();
    State->computeForces();
    State->getForces(force);

    //update second half of velocity vector based on new forces
    ArrayHandle<Dscalar2> h_f(force);
    ArrayHandle<Dscalar2> h_v(velocity);
    for (int i = 0; i < N; ++i)
        {
        h_v.data[i].x += 0.5*deltaT*h_f.data[i].x;
        h_v.data[i].y += 0.5*deltaT*h_f.data[i].y;
        };
    };

/*!
 * Perform a FIRE minimization step on the GPU
 */
template <class T>
void EnergyMinimizerFIRE<T>::fireStepGPU()
    {
    };

/*!
 * Perform a FIRE minimization step on the CPU
 */
template <class T>
void EnergyMinimizerFIRE<T>::fireStepCPU()
    {
    Power = 0.0;
    forceMax = 0.0;
    if (true)//scope for array handles
        {
        //calculate the power, and precompute norms of vectors
        ArrayHandle<Dscalar2> h_f(force);
        ArrayHandle<Dscalar2> h_v(velocity);
        Dscalar forceNorm = 0.0;
        Dscalar velocityNorm = 0.0;
        for (int i = 0; i < N; ++i)
            {
            Power += dot(h_f.data[i],h_v.data[i]);
            Dscalar fdot = dot(h_f.data[i],h_f.data[i]);
            if (fdot > forceMax) forceMax = fdot;
            forceNorm += fdot;
            velocityNorm += dot(h_v.data[i],h_v.data[i]);
            };
        Dscalar scaling = 0.0;
        if(forceNorm > 0.)
            scaling = sqrt(velocityNorm/forceNorm);
        //adjust the velocity according to the FIRE algorithm
        for (int i = 0; i < N; ++i)
            {
            h_v.data[i].x = (1.0-alpha)*h_v.data[i].x + alpha*scaling*h_f.data[i].x;
            h_v.data[i].y = (1.0-alpha)*h_v.data[i].y + alpha*scaling*h_f.data[i].y;
            };
        };

    if (Power > 0)
        {
        if (NSinceNegativePower > NMin)
            {
            deltaT = min(deltaT*deltaTInc,deltaTMax);
            alpha = alpha * alphaDec;
            };
        NSinceNegativePower += 1;
        }
    else
        {
        deltaT = deltaT*deltaTDec;
        alpha = alphaStart;
        ArrayHandle<Dscalar2> h_v(velocity);
        for (int i = 0; i < N; ++i)
            {
            h_v.data[i].x = 0.0;
            h_v.data[i].y = 0.0;
            };
        };
    printf("step %i max force:%f \tpower: %f\n",iterations,forceMax,Power);
    };

/*!
 * Perform a FIRE minimization step on the CPU
 */
template <class T>
void EnergyMinimizerFIRE<T>::minimize()
    {
    //initialized the forces?
    State->computeForces();
    State->getForces(force);
    while( (iterations < maxIterations) && (forceMax > forceCutoff) )
        {
        iterations +=1;
        velocityVerlet();
        fireStep();
        };

    };

#endif

