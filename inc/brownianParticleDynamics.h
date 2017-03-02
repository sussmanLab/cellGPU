#ifndef brownianParticleDynamics_H
#define brownianParticleDynamics_H

#include "simpleEquationOfMotion.h"

/*! \file brownianParticleDynamics.h */
//!A class that implements simple Brownian particle dynamics in 2D
/*!
implements dr/dt = mu*F + sqrt(KT mu/2) R(T), where mu is the inverse drag, KT is the temperature,
and R(T) is a delta-correlated Gaussian noise.
Noise is defined in two dimensions so that the short time diffusion constant is D = KT mu
*/
class brownianParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        brownianParticleDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;Reproducible = false;};

        //!additionally set the number of particles andinitialize things
        brownianParticleDynamics(int N);

        //!the fundamental function that models will call
        virtual void integrateEquationsOfMotion(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);


        //!Get temperature, T
        Dscalar getT(){return Temperature;};
        //!Set temperature, T
        void setT(Dscalar _T){Temperature=_T;};
        //!Get the inverse friction constant, mu
        Dscalar getMu(){return mu;};
        //!Set the value of the inverse friction coefficient
        void setMu(Dscalar _mu){mu=_mu;};

        //!allow for whatever RNG initialization is needed
        virtual void initializeRNGs(int globalSeed, int tempSeed);
        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting(const vector<int> &reIndexer);

    protected:
        //!The temperature. That right there is an A-plus level doxygen description
        Dscalar Temperature;
        //!The value of the inverse friction constant
        Dscalar mu;
        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> RNGs;

    };
#endif
