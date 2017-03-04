#define ENABLE_CUDA

#include "Simulation.h"
/*! \file Simulation.cpp */

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

