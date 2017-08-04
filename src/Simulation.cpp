#define ENABLE_CUDA

#include "Simulation.h"
/*! \file Simulation.cpp */

/*!
Initialize all of the shared points, set default values of things
*/
Simulation::Simulation(): integerTimestep(0), Time(0.),integrationTimestep(0.01),spatialSortThisStep(false),
sortPeriod(-1)
    {
    Box = make_shared<gpubox>();
    };

/*!
Set a pointer to the equation of motion, and give the equation of motion a reference to the
model... this function will be refactored eventually now that EOM are just updaters
*/
void Simulation::setEquationOfMotion(EOMPtr _eom, ForcePtr _config)
    {
    equationOfMotion = _eom;
    _eom->set2DModel(_config);
    updaters.push_back(equationOfMotion);
    };

/*!
Set a pointer to the configuratione, and give the configurationa reference to this
*/
void Simulation::setConfiguration(ForcePtr _config)
    {
    cellConfiguration = _config;
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setCurrentTime(Dscalar _cTime)
    {
    Time = _cTime;
    auto cellConf = cellConfiguration.lock();
    cellConf->setTime(Time);
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setIntegrationTimestep(Dscalar dt)
    {
    integrationTimestep = dt;
    auto cellConf = cellConfiguration.lock();
    auto eom = equationOfMotion.lock();
    cellConf->setDeltaT(dt);
    eom->setDeltaT(dt);
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setCPUOperation(bool setcpu)
    {
    auto cellConf = cellConfiguration.lock();
    auto eom = equationOfMotion.lock();
    if (setcpu)
        {
        cellConf->setCPU();
        eom->setCPU();
        USE_GPU = false;
        }
    else
        {
        cellConf->setGPU();
        eom->setGPU();
        USE_GPU = true;
        };
    };

/*!
\pre the equation of motion already knows if the GPU will be used
\post the dynamics are set to be reproducible if the boolean is true, otherwise the RNG is initialized
*/
void Simulation::setReproducible(bool reproducible)
    {
    auto eom = equationOfMotion.lock();
    if (reproducible)
        eom->setReproducible(true);
    else
        eom->setReproducible(false);
    };

/*!
Calls the configuration to displace the degrees of freedom
*/
void Simulation::computeForces(GPUArray<Dscalar2> &forces)
    {
    auto forceComputer = cellConfiguration.lock();
    forceComputer->computeForces();
    forces.swap(forceComputer->returnForces());
    };

/*!
Calls the configuration to displace the degrees of freedom
*/
void Simulation::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements)
    {
    auto cellConf = cellConfiguration.lock();
    cellConf->moveDegreesOfFreedom(displacements);
    };

/*!
Call all relevant functions to advance the system one time step; every sortPeriod also call the
spatial sorting routine.
\post The simulation is advanced one time step
*/
void Simulation::performTimestep()
    {
    integerTimestep += 1;
    Time += integrationTimestep;
    
    //perform any updates, one of which should probably be an EOM
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->Update(integerTimestep);
        };

    //spatially sort as necessary
    auto cellConf = cellConfiguration.lock();
    //check if spatial sorting needs to occur
    if (sortPeriod > 0 && integerTimestep % sortPeriod == 0)
        {
        cellConf->spatialSorting();
        for (int u = 0; u < updaters.size(); ++u)
            {
            auto upd = updaters[u].lock();
            upd->spatialSorting();
            };
        };
    cellConf->setTime(Time);
    };
