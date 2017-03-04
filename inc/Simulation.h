#ifndef SIMULATION_H
#define SIMULATION_H

#include "Simple2DModel.h"
#include "simpleEquationOfMotion.h"
#include "gpubox.h"
#include "cellListGPU.h"

/*! \file Simulation.h */
//! A class that ties together all the parts of a simulation
/*!
*/
class Simulation
    {
    public:
        //!The equation of motion to run
        simpleEquationOfMotion *equationOfMotion;
        //!pass in an equation of motion to run
        void setEquationOfMotion(simpleEquationOfMotion &_eom){equationOfMotion = &_eom;};

        //!The configuration of cells
        Simple2DModel *cellConfiguration;
        //!Pass in a reference to the configuration
        void setConfiguration(Simple2DModel &_config){cellConfiguration = &_config;};

        //!The domain of the simulation
        gpubox *Box;
        //!Pass in a reference to the box
        void setBox(gpubox &_box){Box = &_box;};

        //!A neighbor list assisting the simulation
        cellListGPU *cellList;;
        //!Pass in a reference to the box
        void setCellList(cellListGPU &_cl){cellList = &_cl;};

        //!Set the simulation timestep
        void setIntegrationTimestep(Dscalar dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        
        int integerTimestep;
        Dscalar Time;
        Dscalar integrationTimestep;

    };
#endif
