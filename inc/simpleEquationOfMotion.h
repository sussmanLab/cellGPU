#ifndef simpleEquationOfMotion_H
#define simpleEquationOfMotion_H

#include "std_include.h"
#include "gpuarray.h"
#include "curand.h"
#include "curand_kernel.h"

/*! \file simpleEquationOfMotion.h
In cellGPU a "simple" equation of motion is one that can take a GPUArray of forces and return a set
of displacements. A derived class of this might be the self-propelled particle equations of motion,
or simple Brownian dynamics.
Derived classes must implement the integrateEquationsOfMotion function
*/
//!A base class for implementing simple equations of motion
class simpleEquationOfMotion
    {
    public:
        //!base constructor sets default time step size
        simpleEquationOfMotion(){deltaT = 0.01; GPUcompute =true;Timestep = 0;Reproducible = false;};

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo, vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> >&IntArrayInfo, GPUArray<Dscalar2> &displacements){};


        //!the fundamental function that models will call to advance the simulation...sometimes the function signature is so simple that this specialization helps
        virtual void integrateEquationsOfMotion(GPUArray<Dscalar2> &forces, GPUArray<Dscalar2> &displacements){};
        //!allow for spatial sorting to be called if necessary... models should pass the "itt" vector to this function
        virtual void spatialSorting(const vector<int> &reIndexer){};
        //!allow for whatever RNG initialization is needed
        virtual void initializeRNGs(int globalSeed, int tempSeed){};

        //!get the number of timesteps run
        int getTimestep(){return Timestep;};
        //!get the current simulation time
        Dscalar getTime(){return (Dscalar)Timestep * deltaT;};
        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};
        //!Get the number of degrees of freedom of the equation of motion
        int getNdof(){return Ndof;};
        //!Set the number of degrees of freedom of the equation of motion
        void setNdof(int _n){Ndof = _n;};
        //!Set whether the integration of the equations of motion should always use the same random numbers
        void setReproducible(bool rep){Reproducible = rep;};

        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU(){GPUcompute = true;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};

        void initializeGPU(bool initGPU){initializeGPURNG = initGPU;};

    protected:
        //!Should the simulation be reproducible (v/v the random numbers generated)?
        bool Reproducible;
        //!A flag to determine whether the CUDA RNGs should be initialized or not (so that the program will run on systems with no GPU by setting this to false
        bool initializeGPURNG;
        //!The number of degrees of freedom the equations of motion need to know about
        int Ndof;
        //! Count the number of integration timesteps
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;
        //!whether the equation of motion does its work on the GPU or not
        bool GPUcompute;
        //!a vector of the re-indexing information
        vector<int> reIndexing;

        //!re-index the any RNGs associated with the e.o.m.
        void reIndexRNG(GPUArray<curandState> &array)
            {
            GPUArray<curandState> TEMP = array;
            ArrayHandle<curandState> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<curandState> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < Ndof; ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
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

#endif
