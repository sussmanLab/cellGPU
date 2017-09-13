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
Add a pointer to the list of updaters, and give that updater a reference to the
model...
*/
void Simulation::addUpdater(UpdaterPtr _upd, ForcePtr _config)
    {
    _upd->set2DModel(_config);
    updaters.push_back(_upd);
    };

/*!
Set a new Box for the simulation...This is the function that should be called to propagate a change
in the box dimensions throughout the simulation...By this time the Box pointed to in the Simulation
is the same one pointed to by the BoxPtrs of Simple2DCell (and, in the Voronoi models, by DelaunayLoc
and cellListGPU), so when we modify it the changes will run through the rest of the simulation
components
*/
void Simulation::setBox(BoxPtr _box)
    {
        //here, instead, get the elements of the BoxPtr and set the contents of Box according to _box's elements... possibly propagate this change throughout
    Dscalar b11,b12,b21,b22;
    _box->getBoxDims(b11,b12,b21,b22);
    if (_box->isBoxSquare())
        Box->setSquare(b11,b22);
    else
        Box->setGeneral(b11,b12,b21,b22);
    };

/*!
Set a pointer to the configuratione, and give the configurationa reference to this
*/
void Simulation::setConfiguration(ForcePtr _config)
    {
    cellConfiguration = _config;
    Box = _config->Box;
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
    cellConf->setDeltaT(dt);
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->setDeltaT(dt);
        };
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void Simulation::setCPUOperation(bool setcpu)
    {
    auto cellConf = cellConfiguration.lock();
    if (setcpu)
        {
        cellConf->setCPU();
        USE_GPU = false;
        }
    else
        {
        cellConf->setGPU();
        USE_GPU = true;
        };

    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        if (setcpu)
            upd->setCPU();
        else
            upd->setGPU();
        };
    };

/*!
\pre the updaters already know if the GPU will be used
\post the updaters are set to be reproducible if the boolean is true, otherwise the RNG is initialized
*/
void Simulation::setReproducible(bool reproducible)
    {
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->setReproducible(reproducible);
        };
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
