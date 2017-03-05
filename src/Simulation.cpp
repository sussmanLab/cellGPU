#define ENABLE_CUDA

#include "Simulation.h"
/*! \file Simulation.cpp */

/*!
Initialize all of the shared points, set default values of things
*/
Simulation::Simulation(): integerTimestep(0), Time(0.),integrationTimestep(0.01)
    {
    Box = make_shared<gpubox>();
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setIntegrationTimestep(Dscalar dt)
    {
    integrationTimestep = dt;
    //cellConfiguration->setDeltaT(dt);
    equationOfMotion->setDeltaT(dt);
    };
/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setCPUOperation(bool setcpu)
    {
    if (setcpu)
        {
        cellConfiguration->setCPU();
        equationOfMotion->setCPU();
        }
    else
        {
        cellConfiguration->setGPU();
        equationOfMotion->setGPU();
        };
    };

