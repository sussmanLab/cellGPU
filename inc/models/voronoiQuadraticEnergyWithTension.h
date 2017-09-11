#ifndef VoronoiQuadraticEnergyWithTension_H
#define VoronoiQuadraticEnergyWithTension_H

#include "voronoiQuadraticEnergy.h"

/*! \file voronoiQuadraticEnergyWithTension.h */
//!Add line tension terms between different "types" of cells in the 2D Voronoi model using kernels in \ref spvKernels
/*!
A child class of VoronoiQuadraticEnergy, this implements an Voronoi model in 2D that can include tension terms between
different types of cells. Different routines are called depending on whether multiple different
cell-cell surface tension values are needed. This specialization exists because on the GPU using the
more generic routine has many more costly memory look-ups, so if it isn't needed the simpler algorithm
should be used.
 */
class VoronoiQuadraticEnergyWithTension : public VoronoiQuadraticEnergy
    {
    public:
        //!initialize with random positions in a square box
        VoronoiQuadraticEnergyWithTension(int n,bool reprod = false) : VoronoiQuadraticEnergy(n,reprod){gamma = 0.0; Tension = false;simpleTension = true;};
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VoronoiQuadraticEnergyWithTension(int n, Dscalar A0, Dscalar P0,bool reprod = false) : VoronoiQuadraticEnergy(n,A0,P0,reprod){gamma = 0;Tension = false;simpleTension = true;};

        //!compute the geometry and get the forces
        virtual void computeForces();
        
        //!compute the quadratic energy functional
        virtual Dscalar computeEnergy();

        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();

        //!Compute the net force on particle i on the CPU with only a single tension value
        virtual void computeVoronoiSimpleTensionForceCPU(int i);

        //!call gpu_force_sets kernel caller
        virtual void computeVoronoiSimpleTensionForceSetsGPU();
        //!Compute the net force on particle i on the CPU with multiple tension values
        virtual void computeVoronoiTensionForceCPU(int i);
        //!call gpu_force_sets kernel caller
        virtual void computeVoronoiTensionForceSetsGPU();

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
