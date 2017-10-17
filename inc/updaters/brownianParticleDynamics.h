#ifndef brownianParticleDynamics_H
#define brownianParticleDynamics_H

#include "simpleEquationOfMotion.h"
#include "Simple2DCell.h"

/*! \file brownianParticleDynamics.h */
//!A class that implements simple Brownian particle dynamics in 2D
/*!
implements \Delta r  = mu*F\Delta t + sqrt(2 T \mu \Delta t ) R(T), where mu is the inverse drag, KT is the temperature,
and R(T) is a delta-correlated Gaussian noise.
Noise is defined in two dimensions so that the short time diffusion constant is D = KT mu
*/
class brownianParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        brownianParticleDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;};
        //!additionally set the number of particles and initialize things
        brownianParticleDynamics(int N);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();

        //!Get temperature, T
        Dscalar getT(){return Temperature;};
        //!Set temperature, T
        void setT(Dscalar _T){Temperature=_T;};
        //!Get the inverse friction constant, mu
        Dscalar getMu(){return mu;};
        //!Set the value of the inverse friction coefficient
        void setMu(Dscalar _mu){mu=_mu;};
        
        //! virtual function to allow the model to be a derived class
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);

        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting(const vector<int> &reIndexer);

    protected:
        //!A shared pointer to a simple cell model
        shared_ptr<Simple2DCell> cellModel;
        //!The temperature. That right there is an A-plus level doxygen description
        Dscalar Temperature;
        //!The value of the inverse friction constant
        Dscalar mu;
    };
#endif
