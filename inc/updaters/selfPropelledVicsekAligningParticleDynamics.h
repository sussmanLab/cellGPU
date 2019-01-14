#ifndef selfPropelledVicsekAligningParticleDynamics_H
#define selfPropelledVicsekAligningParticleDynamics_H

#include "simpleEquationOfMotion.h"
#include "Simple2DActiveCell.h"

/*! \file selfPropelledVicsekAligingParticleDynamics.h */
//!A class that implements simple self-propelled particle dynamics in 2D
/*!
implements dr/dt = mu*F + v_0 \hat{n}, where \hat{n} = (cos(theta),sin(theta))

Additionally there is a tendancy for the directors to align with the directors of
nearby particles
theta_j(t+\Delta t) = arg (\sum_{neighbors of j} (e^(i*theta_k(t)) + \eta*n*e^(i*\xi_j^t)  ),
where k denotes a neighbor of j, where n is the number of neighbors of j, \xi is a random number between 0 and 2Pi
*/
class selfPropelledVicsekAligningParticleDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        selfPropelledVicsekAligningParticleDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;};

        //!additionally set the number of particles andinitialize things
        selfPropelledVicsekAligningParticleDynamics(int N, Dscalar _eta = 0.0, Dscalar _tau = 1.0);

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
        void setEta(Dscalar _Eta){Eta=_Eta;};
        void setTau(Dscalar _tau){tau=_tau;};

        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting();
        //!set the active model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);

    protected:
        //!A shared pointer to a simple active model
        shared_ptr<Simple2DActiveCell> activeModel;
        //!The value of the alignment coupling
        Dscalar tau;
        //!The value of the inverse friction constant
        Dscalar mu;
        //!The value of the strength of vectorial noise
        Dscalar Eta;
    };
#endif
