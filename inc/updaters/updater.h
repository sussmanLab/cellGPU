#ifndef updater_H
#define updater_H

#include "std_include.h"
#include "Simple2DModel.h"

/*! \file updater.h */
//!A base class for implementing simple updaters
/*!
In cellGPU an updater is some class object that can update something about the
underlying state of the system. An example might be an equation of motion, or an updater that periodically subtracts off
any center-of-mass motion of a system as it evolves, etc.. A simulation will call all updaters in a loop,
e.g. for(each updater i in list) updater[i].Update(Timestep)
To facilitate this structure, but acknowledge that any given updater might only need to be called
occasionally, the Update function is passed a timestep, and each updaters has a period that should be set.
*/
class updater
    {
    public:
        //! by default, updaters are called every timestep with no offset
        updater(){Period = -1;Phase = 0;};
        updater(int _p){Period = _p; Phase = 0;};
        //! The fundamental function that a controlling Simulation can call
        virtual void Update(int timestep)
            {
            if(Period >0 && (timestep+Phase) % Period == 0)
                performUpdate();
            };
        //! The function which performs the update
        virtual void performUpdate(){};
        //! A pointer to a Simple2DModel that the updater acts on
        shared_ptr<Simple2DModel> model;
        //! virtual function to allow the model to be a derived class
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){model=_model;};
        //! set the period
        void setPeriod(int _p){Period = _p;};
        //! set the phase
        void setPhase(int _p){Phase = _p;};

        //!allow for spatial sorting to be called if necessary...
        virtual void spatialSorting(){};

        //!Allow for a reproducibility call to be made
        virtual void setReproducible(bool rep){};

        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU(){GPUcompute = true;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};
        //!Get the number of degrees of freedom of the equation of motion
        int getNdof(){return Ndof;};
        //!Set the number of degrees of freedom of the equation of motion
        void setNdof(int _n){Ndof = _n;};

        //!allow all updaters to potentially implement an internal time scale
        virtual void setDeltaT(Dscalar dt){};

    protected:
        //!The period of the updater... the updater will work every Period timesteps
        int Period;
        //!The phase of the updater... the updater will work every Period timesteps offset by a phase
        int Phase;
        //!whether the updater does its work on the GPU or not
        bool GPUcompute;
        //!The number of degrees of freedom the equations of motion need to know about
        int Ndof;
        //!a vector of the re-indexing information
        vector<int> reIndexing;
        //!Re-index cell arrays after a spatial sorting has occured.
        void reIndexArray(GPUArray<int> &array)
            {
            GPUArray<int> TEMP = array;
            ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar> &array)
            {
            GPUArray<Dscalar> TEMP = array;
            ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar2> &array)
            {
            GPUArray<Dscalar2> TEMP = array;
            ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
    };

typedef shared_ptr<updater> UpdaterPtr;
typedef weak_ptr<updater> WeakUpdaterPtr;
#endif
