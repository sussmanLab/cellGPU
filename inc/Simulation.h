#ifndef SIMULATION_H
#define SIMULATION_H

#include "Simple2DCell.h"
#include "simpleEquationOfMotion.h"
#include "gpubox.h"
#include "cellListGPU.h"

/*! \file Simulation.h */

//! A class that ties together all the parts of a simulation
/*!
Simulation objects should have a configuration set, and then an equation of motion. In addition to
being a centralized object controlling the progression of a simulation of cell models, the Simulation
class provides some interfaces to cell configuration and equation of motion parameter setters.
*/
class Simulation : public enable_shared_from_this<Simulation>
    {
    public:
        //!Initialize all the shared pointers, etc.
        Simulation();
        //!pass in an equation of motion to run
        void setEquationOfMotion(EOMPtr _eom,ForcePtr _config);

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

        //!Set the time between spatial sorting operations.
        void setSortPeriod(int sp){sortPeriod = sp;};

        //!reset the simulation clock
        virtual void setCurrentTimestep(int _cTime){integerTimestep =_cTime;};
        //! An integer that keeps track of how often performTimestep has been called
        int integerTimestep;
        //!The current simulation time
        Dscalar Time;
        //! The dt of a time step
        Dscalar integrationTimestep;
        //! A flag controlling whether to use the GPU
        bool USE_GPU;

    protected:
        //! Determines how frequently the spatial sorter be called...once per sortPeriod Timesteps. When sortPeriod < 0 no sorting occurs
        int sortPeriod;
        //!A flag that determins if a spatial sorting is due to occur this Timestep
        bool spatialSortThisStep;

    };
typedef shared_ptr<Simulation> SimulationPtr;
#endif
