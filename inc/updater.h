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
        virtual void set2DModel(shared_ptr<Simple2DModel> _model){};
        //! set the period
        void setPeriod(int _p){Period = _p;};
        //! set the phase
        void setPhase(int _p){Phase = _p;};

        //!allow for spatial sorting to be called if necessary...
        virtual void spatialSorting(){};
    
        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU(){GPUcompute = true;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};

    
    protected:
        //!The period of the updater... the updater will work every Period timesteps
        int Period;
        //!The phase of the updater... the updater will work every Period timesteps offset by a phase
        int Phase;
        //!whether the updater does its work on the GPU or not
        bool GPUcompute;
    };

typedef shared_ptr<updater> UpdaterPtr;
typedef weak_ptr<updater> WeakUpdaterPtr;
#endif
