#ifndef updaterWithNoise_H
#define updaterWithNoise_H

#include "updater.h"
#include "noiseSource.h"



/*! \file updaterWithNoise.h */
//! an updater with a noise source

class updaterWithNoise : public updater
    {
    public:
        //!updaterWithNoise constructor
        updaterWithNoise(){};
        //!updaterWithNoise constructor
        updaterWithNoise(bool rep)
            {
            setReproducible(rep);
            };
        //!Set whether the source of noise should always use the same random numbers
        virtual void setReproducible(bool rep)
            {
            noise.setReproducible(rep);
            if (GPUcompute)
                noise.initializeGPURNGs(1337,0);
            };
        //!re-index the any RNGs associated with the e.o.m.
        void reIndexRNG(GPUArray<curandState> &array)
            {
            GPUArray<curandState> TEMP = array;
            ArrayHandle<curandState> temp(TEMP,access_location::host,access_mode::read);
            ArrayHandle<curandState> ar(array,access_location::host,access_mode::readwrite);
            for (int ii = 0; ii < reIndexing.size(); ++ii)
                {
                ar.data[ii] = temp.data[reIndexing[ii]];
                };
            };
    protected:
        //! A source of noise for the equation of motion
        noiseSource noise;
    };



#endif
