#ifndef SIMULATION_H
#define SIMULATION_H

#include "std_include.h"
#include "Simple2DCell.h"
#include "simpleEquationOfMotion.h"
#include "updater.h"
#include "periodicBoundaries.h"

/*! \file Simulation.h */

//! A class that ties together all the parts of a simulation
/*!
Simulation objects should have a configuration set, and then at least one updater (such as an equation of motion). In addition to
being a centralized object controlling the progression of a simulation of cell models, the Simulation
class provides some interfaces to cell configuration and updater parameter setters.
*/
class Simulation : public enable_shared_from_this<Simulation>
    {
    public:
        //!Initialize all the shared pointers, etc.
        Simulation();
        //!Pass in a reference to the configuration
        void setConfiguration(ForcePtr _config);

        //!Call the force computer to compute the forces
        void computeForces(GPUArray<double2> &forces);
        //!Call the configuration to move particles around
        void moveDegreesOfFreedom(GPUArray<double2> &displacements);
        //!Call every updater to advance one time step
        void performTimestep();

        //!return a shared pointer to this Simulation
        shared_ptr<Simulation> getPointer(){ return shared_from_this();};
        //!The configuration of cells
        WeakForcePtr cellConfiguration;
        //! A vector of updaters that the simulation will loop through
        vector<WeakUpdaterPtr> updaters;

        //!Add an updater
        void addUpdater(UpdaterPtr _upd){updaters.push_back(_upd);};
        //!Add an updater with a reference to a configuration
        void addUpdater(UpdaterPtr _upd, ForcePtr _config);

        //!Clear out the vector of updaters
        void clearUpdaters(){updaters.clear();};

        //!The domain of the simulation
        PeriodicBoxPtr Box;
        //!This changes the contents of the Box pointed to by Box to match that of _box
        void setBox(PeriodicBoxPtr _box);

        //!Set the simulation timestep
        void setIntegrationTimestep(double dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        //!Enforce reproducible dynamics
        void setReproducible(bool reproducible);

        //!Set the time between spatial sorting operations.
        void setSortPeriod(int sp){sortPeriod = sp;};

        //!reset the simulation clock
        virtual void setCurrentTime(double _cTime);
        //!reset the simulation clock counter
        virtual void setCurrentTimestep(int _cTime){integerTimestep =_cTime;};
        //! An integer that keeps track of how often performTimestep has been called
        int integerTimestep;
        //!The current simulation time
        double Time;
        //! The dt of a time step
        double integrationTimestep;
        //! A flag controlling whether to use the GPU
        bool USE_GPU;

        //!Allow a openMP threads
        int ompThreadNum = 1;
        //set number of threads
        virtual void setOmpThreads(int _number);

    protected:
        //! Determines how frequently the spatial sorter be called...once per sortPeriod Timesteps. When sortPeriod < 0 no sorting occurs
        int sortPeriod;
        //!A flag that determins if a spatial sorting is due to occur this Timestep
        bool spatialSortThisStep;

    };
typedef shared_ptr<Simulation> SimulationPtr;
#endif
