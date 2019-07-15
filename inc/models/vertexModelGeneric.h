#ifndef vertexModelGeneric_H
#define vertexModelGeneric_H

#include "vertexModelGenericBase.h"

/*! \file vertexModelGeneric.h */
//! Builds a "maintain topology" on top of vertexModelGenericBase


class vertexModelGeneric : public vertexModelGenericBase
    {
    public:

        //!update/enforce the topology, performing simple T1 transitions
        virtual void enforceTopology();
        virtual void enforceTopologyGPU();
    /*
    //perhaps define helper functions?
        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();
    */
    };
#endif
