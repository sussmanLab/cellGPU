#ifndef selfPropelledParticleDynamics_H
#define selfPropelledParticleDynamics_H

#include "simpleEquationOfMotion.h"

/*! \file selfPropelledParticleDynamics.h */
//!A class that implements simple self-propelled particle dynamics in 2D
/*!
implements dr/dt = mu*F + v_0 \hat{n}, where \hat{n} = (cos(theta),sin(theta)), and
d theta/dt = (brownian noise)
*/
class selfPropelledParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        selfPropelledParticleDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;Reproducible = false;};

        //!additionally set the number of particles andinitialize things
        selfPropelledParticleDynamics(int N);

        //!the fundamental function that models will call
        virtual void integrateEquationsOfMotion(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements);


        //!Get the inverse friction constant, mu
        Dscalar getMu(){return mu;};
        //!Set the number of degrees of freedom of the equation of motion
        void setMu(Dscalar _mu){mu=_mu;};

        //!allow for whatever RNG initialization is needed
        virtual void initializeGPURNGs(int globalSeed=1337, int tempSeed=0);
        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting(const vector<int> &reIndexer);

    protected:
        //!The value of the inverse friction constant
        Dscalar mu;
        //!An array of random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> RNGs;

    };
#endif
