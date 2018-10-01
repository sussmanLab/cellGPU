#ifndef selfPropelledAligningParticleDynamics_H
#define selfPropelledAligningParticleDynamics_H

#include "simpleEquationOfMotion.h"
#include "Simple2DActiveCell.h"

/*! \file selfPropelledAligingParticleDynamics.h */
//!A class that implements simple self-propelled particle dynamics in 2D
/*!
implements dr/dt = mu*F + v_0 \hat{n}, where \hat{n} = (cos(theta),sin(theta))

Additionally there is a tendancy for the directors to align with the particles
current instantaneous velocity,
theta/dt = -J sin(\theta_i-\phi_i)+(brownian noise), where \phi_i = dr/dt

*/
class selfPropelledAligningParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        selfPropelledAligningParticleDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;};

        //!additionally set the number of particles andinitialize things
        selfPropelledAligningParticleDynamics(int N);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();

        //!Get the inverse friction constant, mu
        Dscalar getMu(){return mu;};
        //!Set the number of degrees of freedom of the equation of motion
        void setMu(Dscalar _mu){mu=_mu;};
        void setJ(Dscalar _J){J=_J;};

        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting();
        //!set the active model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);

    protected:
        //!A shared pointer to a simple active model
        shared_ptr<Simple2DActiveCell> activeModel;
        //!The value of the inverse friction constant
        Dscalar mu;
        //!The value of the aligning couping
        Dscalar J;
    };
#endif
