#ifndef VoronoiQuadraticEnergy_H
#define VoronoiQuadraticEnergy_H

#include "voronoiModelBase.h"
#include "voronoiQuadraticEnergy.cuh"

/*! \file voronoiQuadraticEnergy.h */
//!Implement a 2D Voronoi model, with and without some extra bells and whistles, using kernels in \ref spvKernels
/*!
 *A child class of voronoiModelBase, this implements a Voronoi model in 2D. This involves mostly calculating
  the forces in the Voronoi model. Optimizing these procedures for
  hybrid CPU/GPU-based computation involves declaring and maintaining several related auxiliary
  data structures that capture different features of the local topology and local geoemetry for each
  cell.
 */
class VoronoiQuadraticEnergy : public voronoiModelBase
    {
    public:
        //!initialize with random positions in a square box
        VoronoiQuadraticEnergy(int n,bool reprod = false);
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VoronoiQuadraticEnergy(int n, Dscalar A0, Dscalar P0,bool reprod = false);
        //!Blank constructor
        VoronoiQuadraticEnergy(){};

        //!Initialize voronoiQuadraticEnergy and call the initializer chain
        void initializeVoronoiQuadraticEnergy(int n);

        //!compute the geometry and get the forces
        virtual void computeForces();

        //!compute the quadratic energy functional
        virtual Dscalar computeEnergy();

        //cell-dynamics related functions...these call functions in the next section
        //in general, these functions are the common calls, and test flags to know whether to call specific versions of specialty functions

        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();
        //!Add up the force sets to get the net force per particle on the GPU
        void SumForcesGPU();

        //CPU functions
        //!Compute the net force on particle i on the CPU
        virtual void computeVoronoiForceCPU(int i);

        //GPU functions
        //!call gpu_force_sets kernel caller
        virtual void computeVoronoiForceSetsGPU();
        //! call gpu_sum_force_sets kernel caller
        void sumForceSets();
        //!call gpu_sum_force_sets_with_exclusions kernel caller
        void sumForceSetsWithExclusions();

        //!Report various cell infor for testing and debugging
        void reportCellInfo();
        //!Report information about net forces...
        void reportForces(bool verbose);

        //!Save tuples for half of the dynamical matrix
        virtual void getDynMatEntries(vector<int2> &rcs, vector<Dscalar> &vals,Dscalar unstress = 1.0, Dscalar stress = 1.0);

        //!calculate the current global off-diagonal stress
        virtual Dscalar getSigmaXY();

    protected:
        //! Second derivative of the energy w/r/t cell positions...for getting dynMat info
        Matrix2x2 d2Edridrj(int i, int j, neighborType neighbor,Dscalar unstress = 1.0, Dscalar stress = 1.0);

    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    };

#endif
