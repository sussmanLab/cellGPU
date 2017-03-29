#ifndef SPVTENSION2D_H
#define SPVTENSION2D_H

#include "std_include.h"
#include "spv2d.h"

/*! \file spv2dTension.h */
//!Add line tension terms between different "types" of cells in the 2D SPV model using kernels in \ref spvKernels
/*!
A child class of SPV2D, this implements an SPV model in 2D that can include tension terms between
different types of cells. Different routines are called depending on whether multiple different
cell-cell surface tension values are needed. This specialization exists because on the GPU using the
more generic routine has many more costly memory look-ups, so if it isn't needed the simpler algorithm
should be used.
 */
class SPVTension2D : public SPV2D
    {
    public:
        //!initialize with random positions in a square box
        SPVTension2D(int n,bool reprod = false) : SPV2D(n,reprod){gamma = 0.0; Tension = false;simpleTension = true;};
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        SPVTension2D(int n, Dscalar A0, Dscalar P0,bool reprod = false) : SPV2D(n,A0,P0,reprod){gamma = 0;Tension = false;simpleTension = true;};

        //!compute the geometry and get the forces
        virtual void computeForces();

        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();

        //!Compute the net force on particle i on the CPU with only a single tension value
        virtual void computeSPVSimpleTensionForceCPU(int i);

        //!call gpu_force_sets kernel caller
        virtual void computeSPVSimpleTensionForceSetsGPU();
        //!Compute the net force on particle i on the CPU with multiple tension values
        virtual void computeSPVTensionForceCPU(int i);
        //!call gpu_force_sets kernel caller
        virtual void computeSPVTensionForceSetsGPU();

        //!Use surface tension
        void setUseSurfaceTension(bool use_tension){Tension = use_tension;};
        //!Set surface tension, with only a single value of surface tension
        void setSurfaceTension(Dscalar g){gamma = g; simpleTension = true;};
        //!Set a general flattened 2d matrix describing surface tensions between many cell types
        void setSurfaceTension(vector<Dscalar> gammas);
        //!Get surface tension
        Dscalar getSurfaceTension(){return gamma;};
    protected:
        //!The value of surface tension between two cells of different type (some day make this more general)
        Dscalar gamma;
        //!A flag specifying whether the force calculation contains any surface tensions to compute
        bool Tension;
        //!A flag switching between "simple" tensions (only a single value of gamma for every unlike interaction) or not
        bool simpleTension;
        //!A flattened 2d matrix describing the surface tension, \gamma_{i,j} for types i and j
        GPUArray<Dscalar> tensionMatrix;


    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    };

#endif
