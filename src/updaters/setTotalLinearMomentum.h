#ifndef setTotalLinearMomentum_H
#define setTotalLinearMomentum_H

#include "updater.h"

/*! \file setTotalLinearMomemtum.h */
//! Uniformly shifts the velocities to maintain a total target momentum
class setTotalLinearMomentum : public updater
    {
    public:
        //! By default, target zero total linear momentum
        setTotalLinearMomentum(double px = 0.0, double py = 0.0);

        //!sets the target x and y total system momentum
        void setMomentumTarget(double px, double py);

        //! performUpdate just selects either the GPU or CPU branch
        virtual void performUpdate();
        
        //!call the CPU routine to set the total linear momentum
        virtual void setLinearMomentumCPU();
        //!call the GPU routine to set the total linear momentum
        virtual void setLinearMomentumGPU();
    protected:
        //!A helper vector for the GPU branch...can be asked to store m[i]*v[i] as an array
        GPUArray<double2> pArray;
        //!A helper structure for performing parallel reduction of the keArray
        GPUArray<double2> pIntermediateReduction;
        //!Helper structure for the GPU branch. A 2 component arry that contains the target and current linear momentum
        GPUArray<double2> pHelper;

    };
#endif
