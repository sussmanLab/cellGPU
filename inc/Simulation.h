#ifndef SIMULATION_H
#define SIMULATION_H

#include "Simple2DCell.h"
#include "simpleEquationOfMotion.h"
#include "gpubox.h"
#include "cellListGPU.h"

/*! \file Simulation.h */

//! A class that ties together all the parts of a simulation
/*!
*/
class Simulation : public enable_shared_from_this<Simulation>
    {
    public:
        //!Initialize all the shared pointers, etc.
        Simulation();
        //!pass in an equation of motion to run
        void setEquationOfMotion(EOMPtr _eom);

        //!Pass in a reference to the configuration
        void setConfiguration(ForcePtr _config);

        //!Call the force computer to compute the forces
        void computeForces(GPUArray<Dscalar2> &forces);
        //!Call the configuration to move particles around
        void moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements);
        //!Call the equation of motion to advance one time step
        void performTimestep();

        //!return a shared pointer to this Simulation
        shared_ptr<Simulation> getPointer(){ return shared_from_this();};
        //!The equation of motion to run
        WeakEOMPtr equationOfMotion;
        //!The configuration of cells
        WeakForcePtr cellConfiguration;
        //!The domain of the simulation
        BoxPtr Box;
        //!Pass in a reference to the box
        void setBox(BoxPtr _box){Box = _box;};

        //!A neighbor list assisting the simulation
        cellListGPU *cellList;;
        //!Pass in a reference to the box
        void setCellList(cellListGPU &_cl){cellList = &_cl;};

        //!Set the simulation timestep
        void setIntegrationTimestep(Dscalar dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        //!Enforce reproducible dynamics
        void setReproducible(bool reproducible);


        int integerTimestep;
        Dscalar Time;
        Dscalar integrationTimestep;
        bool USE_GPU;

    };
typedef shared_ptr<Simulation> SimulationPtr;
#endif
