#ifndef SPVTENSION2D_H
#define SPVTENSION2D_H

#include "std_include.h"
#include "spv2d.h"

/*! \file spv2dTension.h */
//!Add line tension terms between different "types" of cells in the 2D SPV model using kernels in \ref spvKernels
/*!
A child class of SPV2D, this implements an SPV model in 2D that can include tension terms between
different types of cells. Different routines are called depending on whether multiple different cell-cell surface tension values are needed.
 */
class SPVTension2D : public SPV2D
    {
    public:
        //!initialize with random positions in a square box
        SPVTension2D(int n,bool reprod = false) : SPV2D(n,reprod){gamma = 0.0; Tension = false;};
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        SPVTension2D(int n, Dscalar A0, Dscalar P0,bool reprod = false) : SPV2D(n,A0,P0,reprod){gamma = 0;Tension = false;};

        //!compute the geometry and get the forces
        virtual void computeForces();

        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();

        //!Compute the net force on particle i on the CPU
        virtual void computeSPVTensionForceCPU(int i);
        //!call gpu_force_sets kernel caller
        virtual void computeSPVTensionForceSetsGPU();

        //!Use surface tension
        void setUseSurfaceTension(bool use_tension){Tension = use_tension;};
        //!Set surface tension
        void setSurfaceTension(Dscalar g){gamma = g;};
        //!Get surface tension
        Dscalar getSurfaceTension(){return gamma;};
    protected:
        //!The value of surface tension between two cells of different type (some day make this more general)
        Dscalar gamma;
        bool Tension;

    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    };

#endif
