#ifndef velocityVerlet_H
#define velocityVerlet_H

#include "simpleEquationOfMotion.h"
#include "Simple2DCell.h"
/*! \file velocityVerlet.h */

class velocityVerlet : public simpleEquationOfMotion
    {
    public:
        velocityVerlet(int nPoint, bool  usegpu =true);
        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();
        //! virtual function to allow the model to be a derived class
        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){State = _model;};
    };
#endif
