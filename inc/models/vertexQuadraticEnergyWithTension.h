#ifndef VertexQuadraticEnergyWithTension_H
#define VertexQuadraticEnergyWithTension_H

#include "vertexQuadraticEnergy.h"

/*! \file vertexQuadraticEnergyWithTension.h */
//!Add line tension terms between different "types" of cells in the 2D Vertex model
/*!
This child class of VertexQuadraticEnergy is completely analogous to the voversion on the Voronoi side.
It implements different tension terms between different types of cells, and different routines are
called depending on whether multiple different cell-cell surface tension values are needed.
 */
class VertexQuadraticEnergyWithTension : public VertexQuadraticEnergy
    {
    public:
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VertexQuadraticEnergyWithTension(int n, Dscalar A0, Dscalar P0,bool reprod = false, bool runSPVToInitialize=false) : VertexQuadraticEnergy(n,A0,P0,reprod,runSPVToInitialize){gamma = 0;Tension = false;simpleTension = true;};

        //!compute the geometry and get the forces
        virtual void computeForces();
        
        //!compute the quadratic energy functional
        virtual Dscalar computeEnergy();

        //!Compute the forces on the GPU with only a single tension value
        virtual void computeVertexSimpleTensionForceGPU();
        //!Compute the net force on particle i on the CPU with multiple tension values
        virtual void computeVertexTensionForcesCPU();
        //!call gpu_force_sets kernel caller
        virtual void computeVertexTensionForceGPU();

        //!Use surface tension
        void setUseSurfaceTension(bool use_tension){Tension = use_tension;};
        //!Set surface tension, with only a single value of surface tension
        void setSurfaceTension(Dscalar g){gamma = g; simpleTension = true;};
        //!Set a general flattened 2d matrix describing surface tensions between many cell types
        void setSurfaceTension(vector<Dscalar> gammas);
        //!Get surface tension
        Dscalar getSurfaceTension(){return gamma;};
    protected:
        //!The value of surface tension between two cells of different type 
        Dscalar gamma;
        //!A flag specifying whether the force calculation contains any surface tensions to compute
        bool Tension;
        //!A flag switching between "simple" tensions (only a single value of gamma for every unlike interaction) or not
        bool simpleTension;
        //!A flattened 2d matrix describing the surface tension, \gamma_{i,j} for types i and j
        GPUArray<Dscalar> tensionMatrix;

    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabaseNetCDF;
    };

#endif
