#include "velocityVerlet.h"
#include "EnergyMinimizerFIRE2D.cuh"//FIRE, written first, has by default some velocity verlet functions
/*! \file velocityVerlet.cpp */

velocityVerlet::velocityVerlet(int nPoint, bool  usegpu)
    {
    Timestep = 0;
    deltaT=0.01;
    GPUcompute=usegpu;
    Ndof = nPoint;
    displacements.resize(Ndof);
    };

void velocityVerlet::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (State->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = State->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    };

void velocityVerlet::integrateEquationsOfMotionCPU()
    {
        {//scope for arrayhandles
    ArrayHandle<double2> h_f(State->returnForces());
    ArrayHandle<double2> h_v(State->returnVelocities());
    ArrayHandle<double2> h_d(displacements);
    for (int i = 0; i < Ndof; ++i)
        {
        //update displacement
        h_d.data[i].x = deltaT*h_v.data[i].x+0.5*deltaT*deltaT*h_f.data[i].x;
        h_d.data[i].y = deltaT*h_v.data[i].y+0.5*deltaT*deltaT*h_f.data[i].y;
        //do first half of velocity update
        h_v.data[i].x += 0.5*deltaT*h_f.data[i].x;
        h_v.data[i].y += 0.5*deltaT*h_f.data[i].y;
        };
        };//end arrayhandle scope

    //move particles, then update the forces
    State->moveDegreesOfFreedom(displacements);
    State->enforceTopology();
    State->computeForces();

    //update second half of velocity vector based on new forces
    ArrayHandle<double2> h_f(State->returnForces());
    ArrayHandle<double2> h_v(State->returnVelocities());
    for (int i = 0; i < Ndof; ++i)
        {
        h_v.data[i].x += 0.5*deltaT*h_f.data[i].x;
        h_v.data[i].y += 0.5*deltaT*h_f.data[i].y;
        };
    };

void velocityVerlet::integrateEquationsOfMotionGPU()
    {
    //calculate displacements and update velocities
        {//scope for array handles
    ArrayHandle<double2> d_f(State->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_d(displacements,access_location::device,access_mode::overwrite);
    gpu_displacement_velocity_verlet(d_d.data,d_v.data,d_f.data,deltaT,Ndof);
    gpu_update_velocity(d_v.data,d_f.data,deltaT,Ndof);
        };
    //move particles and update forces
    State->moveDegreesOfFreedom(displacements);
    State->enforceTopology();
    State->computeForces();

    //update velocities over the second half-step
    ArrayHandle<double2> d_f(State->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(State->returnVelocities(),access_location::device,access_mode::readwrite);
    gpu_update_velocity(d_v.data,d_f.data,deltaT,Ndof);
    };
