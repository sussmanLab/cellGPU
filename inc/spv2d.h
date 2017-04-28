#ifndef SPV_H
#define SPV_H

#include "std_include.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

#include "Matrix.h"
#include "functions.h"
#include "DelaunayMD.h"
#include "spv2d.cuh"

/*! \file spv2d.h */
//!Implement the 2D SPV model, with and without some extra bells and whistles, using kernels in \ref spvKernels
/*!
 *A child class of DelaunayMD, this implements the SPV model in 2D. This involves mostly calculating
  the forces in the SPV model and then moving cells appropriately. Optimizing these procedures for
  hybrid CPU/GPU-based computation involves declaring and maintaining several related auxiliary
  data structures that capture different features of the local topology and local geoemetry for each
  cell.
 */
class SPV2D : public DelaunayMD
    {
    public:
        //!initialize with random positions in a square box
        SPV2D(int n,bool reprod = false);
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        SPV2D(int n, Dscalar A0, Dscalar P0,bool reprod = false);
        //!Blank constructor
        SPV2D(){};

        //!Initialize DelaunayMD, set random orientations for cell directors, prepare data structures
        void Initialize(int n);

        //!return the forces
        virtual void getForces(GPUArray<Dscalar2> &forces){forces = cellForces;};

        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<Dscalar2> & returnForces(){return cellForces;};

        //!compute the geometry and get the forces
        virtual void computeForces();

        //!update/enforce the topology
        virtual void enforceTopology();

        //!Declare which particles are to be excluded (exes[i]!=0)
        void setExclusions(vector<int> &exes);

        //cell-dynamics related functions...these call functions in the next section
        //in general, these functions are the common calls, and test flags to know whether to call specific versions of specialty functions

        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();
        //!Add up the force sets to get the net force per particle on the GPU
        void SumForcesGPU();

        //CPU functions
        //!Compute cell geometry on the CPU
        void computeGeometryCPU();
        //!Compute the net force on particle i on the CPU
        virtual void computeSPVForceCPU(int i);

        //GPU functions
        //!call gpu_compute_geometry kernel caller
        void computeGeometryGPU();
        //!call gpu_force_sets kernel caller
        virtual void computeSPVForceSetsGPU();
        //! call gpu_sum_force_sets kernel caller
        void sumForceSets();
        //!call gpu_sum_force_sets_with_exclusions kernel caller
        void sumForceSetsWithExclusions();


        //!Report various cell infor for testing and debugging
        void reportCellInfo();
        //!Report information about net forces...
        void reportForces(bool verbose);

    //protected functions
    protected:
        //! call getDelSets for all particles
        void allDelSets();

        //! Maintain the delSets and delOther data structure for particle i
        //! If it returns false there was a problem and a global re-triangulation is triggered.
        bool getDelSets(int i);
        //!resize all neighMax-related arrays
        void resetLists();
        //!sort points along a Hilbert curve for data locality
        void spatialSorting();


    public:
        //Various partial derivatives related to calculating the dynamical matrix
        //!The derivative of a voronoi vertex position with respect to change in the first cells position
        Matrix2x2 dHdri(Dscalar2 ri, Dscalar2 rj, Dscalar2 rk);

        //!Derivative of the area of cell i with respect to the position of cell j
        Dscalar2 dAidrj(int i, int j);
        //!Derivative of the perimeter of cell i with respect to the position of cell j
        Dscalar2 dPidrj(int i, int j);

        //!second derivatives of voronoi vertex with respect to cell positions
        vector<Dscalar> d2Hdridrj(Dscalar2 rj, Dscalar2 rk, int jj);

        //! Second derivative of the energy w/r/t cell positions
        Matrix2x2 d2Edridrj(int i, int j, neighborType neighbor,Dscalar unstress = 1.0, Dscalar stress = 1.0);
        //!Second derivative of area w/r/t voronoi and cell position
        Matrix2x2 d2Areadvdr(Matrix2x2 &dvpdr, Matrix2x2 &dvmdr);
        //!Second derivative of perimeter w/r/t voronoi and cell position
        Matrix2x2 d2Peridvdr(Matrix2x2 &dvdr, Matrix2x2 &dvmdr, Matrix2x2 &dvpdr,Dscalar2 vm, Dscalar2 v, Dscalar2 vp);

        //!Save tuples for half of the dynamical matrix
        void getDynMatEntries(vector<int2> &rcs, vector<Dscalar> &vals,Dscalar unstress = 1.0, Dscalar stress = 1.0);

    //protected member variables
    protected:
        //!A flag that notifies the existence of any particle exclusions (for which the net force is set to zero by fictitious external forces)
        bool particleExclusions;

        //!delSet.data[n_idx(nn,i)] are the previous and next consecutive delaunay neighbors
        /*! These are orientationally ordered, of point i (for use in computing forces on GPU)
        */
        GPUArray<int2> delSets;
        //!delOther.data[n_idx(nn,i)] contains the index of the "other" delaunay neighbor.
        /*!
        i.e., the mutual neighbor of delSet.data[n_idx(nn,i)].y and delSet.data[n_idx(nn,i)].z that isn't point i
        */
        GPUArray<int> delOther;

        //!In GPU mode, interactions are computed "per voronoi vertex"...forceSets are summed up to get total force on a particle
        GPUArray<Dscalar2> forceSets;

    //public member variables
    public:
        //! Read from a database what the time of the simulation when saved was
        Dscalar SimTime;
        //!"exclusions" zero out the force on a cell...the external force needed to do this is stored in external_forces
        GPUArray<Dscalar2> external_forces;
        //!An array containing the indices of excluded particles
        GPUArray<int> exclusions;

        //!Some function-timing-related scalars
        Dscalar triangletiming, forcetiming;
    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    };

#endif
