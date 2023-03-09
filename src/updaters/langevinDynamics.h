#ifndef langevinDynamics_H
#define langevinDynamics_H

#include "simpleEquationOfMotion.h"
#include "Simple2DCell.h"

/*! \file langevinDynamics.h */
//!A class that implements simple Langevin particle dynamics in 2D
/*!
Implements dq = m^{-1} p dt; dp = -\nabla U(q) dt - \gamma p dt + \sigma \sqrt{m} dW, where the W are wiener processes
\gamma is a free parameter, and \sigma is \sqrt{2\gamma kT}

This implements the so-called BAOAB scheme of Leimkuhler and Matthews (https://doi.org/10.1093/amrx/abs010 and  https://doi.org/10.1063/1.4802990)

Noise is defined in two dimensions so that the short time diffusion constant is D = KT mu
*/
class langevinDynamics : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size
        langevinDynamics(){deltaT = 0.01; GPUcompute =true;Timestep = 0;};
        //!additionally set the number of particles and initialize things
        langevinDynamics(int N, double temperature=1., double gamma=1., bool usegpu=true);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();

        //!Get temperature, T
        double getT(){return Temperature;};
        //!Set temperature, T
        void setT(double _T){Temperature=_T;};
        //!Get the inverse friction constant, mu
        double getGamma(){return gamma;};
        //!Set the value of the inverse friction coefficient
        void setGamma(double _gamma){gamma=_gamma;};

        //! virtual function to allow the model to be a derived class
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);

        //!call the Simple2DCell spatial vertex sorter, and re-index arrays of cell activity
        virtual void spatialSorting(const vector<int> &reIndexer);

    protected:
        //!A shared pointer to a simple cell model
        shared_ptr<Simple2DCell> cellModel;
        //!The temperature. That right there is an A-plus level doxygen description
        double Temperature;
        //!The value of the inverse friction constant
        double gamma;
        //!The implied value of \sigma
        double sigma;
    };
#endif
