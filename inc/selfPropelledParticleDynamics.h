#ifndef selfPropelledParticleDynamics_H
#define selfPropelledParticleDynamics_H

#include "simpleEquationOfMotion.h"

/*! \file selfPropelledParticleDynamics.h
implements dr/dt = mu*F = v_0 \hat{n}, where \hat{n} = (cos(theta),sin(theta)), and 
d theta/dt = (brownian noise)
*/
//!A class that implements simple self-propelled particle dynamics in 2D
class selfPropelledParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets default time step size
        selfPropelledParticleDynamics(int N);
        //!the fundamental function that advances the simulation
        virtual void integrateEquationsOfMotion(GPUArray<Dscalar2> &forces, GPUArray<Dscalar2> &displacements);
        //!call the CPU routine to integrate the e.o.m.
        void integrateEquationsOfMotionCPU(GPUArray<Dscalar2> &forces, GPUArray<Dscalar2> &displacements);
        //!call the GPU routine to integrate the e.o.m.
        void integrateEquationsOfMotionGPU(GPUArray<Dscalar2> &forces, GPUArray<Dscalar2> &displacements);

        //!Get the inverse friction constant, mu
        Dscalar getMu(){return mu;};
        //!Set the number of degrees of freedom of the equation of motion
        void setMu(Dscalar _mu){mu=_mu;};

        //!Set uniform motility
        void setv0Dr(Dscalar v0new,Dscalar drnew);
        //!Set non-uniform cell motilites
        void setCellMotility(vector<Dscalar> &v0s,vector<Dscalar> &drs);
        //!Set random cell directors (for active cell models)
        void setCellDirectorsRandomly();

        //!An array of angles (relative to the x-axis) that the cell directors point
        GPUArray<Dscalar> cellDirectors;
        //!velocity of cells in mono-motile systems
        Dscalar v0;
        //!rotational diffusion of cell directors in mono-motile systems
        Dscalar Dr;
        //!The motility parameters (v0 and Dr) for each cell
        GPUArray<Dscalar2> Motility;
        
        //!allow for whatever RNG initialization is needed
        virtual void initializeRNGs(int globalSeed, int tempSeed);
        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting(const vector<int> &reIndexer);

    protected:
        //!The value of the inverse friction constant
        Dscalar mu;
        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> RNGs;

    };
#endif
