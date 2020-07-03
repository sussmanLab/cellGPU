#include "setTotalLinearMomentum.h"
#include "setTotalLinearMomentum.cuh"
#include "utilities.cuh"
/*! \file setTotalLinearMomentum.cpp */

setTotalLinearMomentum::setTotalLinearMomentum(double px, double py)
    {
    setMomentumTarget(px,py);
    };

void setTotalLinearMomentum::setMomentumTarget(double px, double py)
    {
    pHelper.resize(2);
    ArrayHandle<double2> h_ph(pHelper);
    h_ph.data[0] = make_double2(px,py);
    h_ph.data[1] = make_double2(0,0);
    //the following will be auto-resized once performUpdate is called
    pArray.resize(1);
    pIntermediateReduction.resize(1);
    };

void setTotalLinearMomentum::performUpdate()
    {
    int N = model->getNumberOfDegreesOfFreedom();
    if(N!=pArray.getNumElements())
        {
        pArray.resize(N);
        pIntermediateReduction.resize(N);
        };
    if(GPUcompute)
        setLinearMomentumGPU();
    else
        setLinearMomentumCPU();
    };

/*!
v[i] = v[i] + (1/(N*m[i]))*(Ptarget - Pcurrent)
*/
void setTotalLinearMomentum::setLinearMomentumCPU()
    {
    ArrayHandle<double2> h_ph(pHelper);
    ArrayHandle<double2> h_v(model->returnVelocities());
    ArrayHandle<double>  h_m(model->returnMasses());
    int N = model->getNumberOfDegreesOfFreedom();
    //!update the current P
    h_ph.data[1]  = make_double2(0.0,0.0);
    for (int ii = 0; ii < N; ++ii)
        h_ph.data[1] = h_ph.data[1] + h_m.data[ii]*h_v.data[ii];
    double2 Pshift = make_double2(h_ph.data[0].x-h_ph.data[1].x,h_ph.data[0].y-h_ph.data[1].y);
    for (int ii = 0; ii < N; ++ii)
        {
        h_v.data[ii] = h_v.data[ii]+(1.0/(N*h_m.data[ii]))*Pshift;
        };
    };

/*!
v[i] = v[i] + (1/(N*m[i]))*(Ptarget - Pcurrent)
*/
void setTotalLinearMomentum::setLinearMomentumGPU()
    {
    {//arrayHandle scope to first find the current total lienar momentum
    ArrayHandle<double> d_m(model->returnMasses(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(model->returnVelocities(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_p(pArray,access_location::device,access_mode::overwrite);
    gpu_dot_double_double2_vectors(d_m.data,d_v.data,d_p.data,model->getNumberOfDegreesOfFreedom());
    };
    {//arrayHandle scope for parallel reduction
    ArrayHandle<double2> d_p(pArray,access_location::device,access_mode::read);
    ArrayHandle<double2> d_pIntermediate(pIntermediateReduction,access_location::device,access_mode::overwrite);
    ArrayHandle<double2> d_P(pHelper,access_location::device,access_mode::readwrite);
    gpu_parallel_reduction(d_p.data,d_pIntermediate.data,d_P.data,1,model->getNumberOfDegreesOfFreedom());
    };
    {//finally, shift the velocities around
    ArrayHandle<double2> h_P(pHelper,access_location::host,access_mode::readwrite);
    double2 Pshift = h_P.data[0] - h_P.data[1];
    ArrayHandle<double> d_m(model->returnMasses(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    gpu_shift_momentum(d_v.data,d_m.data,Pshift,model->getNumberOfDegreesOfFreedom());
    };
    };
