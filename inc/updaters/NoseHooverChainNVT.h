#ifndef NoseHooverChainNVT_H
#define NoseHooverChainNVT_H

#include "simpleEquationOfMotion.h"
#include "Simple2DCell.h"

/*! \file NoseHooverChainNVT.h */
//! Implements NVT dynamics according to the Nose-Hoover equations of motion with a chain of thermostats
/*!
 *This allows one to do standard NVT simulations. A chain (whose length can be specified by the user)
 of thermostats is used to maintain the target temperature. We closely follow the Frenkel & Smit
 update scheme, which is itself based on:
 Martyna, Tuckerman, Tobias, and Klein
 Mol. Phys. 87, 1117 (1996)
*/
class NoseHooverChainNVT : public simpleEquationOfMotion
    {
    public:
        //!The base constructor asks for the number of particles and the length of the chain
        NoseHooverChainNVT(int N, int M=2);

        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){State = _model;};

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();

        //!Get temperature, T
        Dscalar getT(){return Temperature;};
        //!Set temperature, T, and also the bath masses!
        void setT(Dscalar _T);

    protected:
        //!The targeted temperature
        Dscalar Temperature;
        //!The length of the NH chain
        int Nchain;
        //!The number of particles in the State
        int Ndof;
        //!the (position,velocity,mass) of the bath degrees of freedom
        GPUArray<Dscalar3> BathVariables;
    };
#endif
