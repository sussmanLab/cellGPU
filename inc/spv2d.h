#ifndef SPV_H
#define SPV_H

#include "std_include.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

#include "Matrix.h"
#include "cu_functions.h"
#include "DelaunayMD.h"
#include "spv2d.cuh"

/*!
 *A child class of DelaunayMD, this implements the SPV model in 2D. This involves mostly calculating
  the forces in the SPV model and then moving cells appropriately. Optimizing these procedures for
  hybrid CPU/GPU-based computation involves declaring and maintaining several related auxiliary
  data structures that capture different features of the local topology and local geoemetry for each
  cell.
 */
 //!Implement the 2D SPV model, with and without some extra bells and whistles, using kernels in \ref spvKernels
class SPV2D : public DelaunayMD
    {
    //public functions
    public:
        //!initialize with random positions in a square box
        SPV2D(int n,bool reprod = false,bool initGPURNG=true);
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        SPV2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true);

        //!Initialize DelaunayMD, set random orientations for cell directors, prepare data structures
        void Initialize(int n,bool initGPU = true);

        //!Set the value of tension to apply between cells of different type (if desired)
        void setTension(Dscalar g){gamma = g;};
        //!Declare that tensions of magnitude gamma should be applied between cells of different type
        void setUseTension(bool u){useTension = u;};
        //!Set all cells to the same "type"
        void setCellTypeUniform(int i);
        //!Set cells to different "type"
        void setCellType(vector<int> &types);
        //!A specialty function for setting cell types within a central ellipse to type 0, and those outside to type 1
        void setCellTypeEllipse(Dscalar frac, Dscalar aspectRatio);
        //!A specialty function for setting cells within a central strip (surface normal to x) to type 0, and others to type 1
        void setCellTypeStrip(Dscalar frac);

        //!Set the time between spatial sorting operations.
        void setSortPeriod(int sp){sortPeriod = sp;};

        //!Declare which particles are to be excluded (exes[i]!=0)
        void setExclusions(vector<int> &exes);

        //! call getDelSets for all particles
        void allDelSets();

        //cell-dynamics related functions...these call functions in the next section
        //in general, these functions are the common calls, and test flags to know whether to call specific versions of specialty functions
        //!Perform a timestep for the system
        void performTimestep();
        //!call the CPU branch to advance the system
        void performTimestepCPU();
        //!call the GPU branch to advance the system
        void performTimestepGPU();

        //!Compute force sets on the GPU
        void ComputeForceSetsGPU();
        //!Add up the force sets to get the net force per particle on the GPU
        void SumForcesGPU();

        //CPU functions
        //!Compute cell geometry on the CPU
        void computeGeometryCPU();
        //!Compute the net force on particle i on the CPU
        void computeSPVForceCPU(int i);
        //!Calculates the displacements and cell director changes on the CPU. Uses a non-reproducible RNG
        void calculateDispCPU();


        //GPU functions
        //!call gpu_displace_and_rotate kernel caller
        void DisplacePointsAndRotate();
        //!call gpu_compute_geometry kernel caller
        void computeGeometryGPU();
        //!call gpu_force_sets kernel caller
        void computeSPVForceSetsGPU();
        //!call gpu_force_sets_tension kernel caller
        void computeSPVForceSetsWithTensionsGPU();
        //! call gpu_sum_force_sets kernel caller
        void sumForceSets();
        //!call gpu_sum_force_sets_with_exclusions kernel caller
        void sumForceSetsWithExclusions();


        //!Report various cell infor for testing and debugging
        void reportCellInfo();
        //!Report information about net forces...for debugging
        void reportForces();
        //!Report the mean net force per particle...better be close to zero!
        void meanForce();
        //!Report the mean area per cell in the system
        void meanArea();

    //protected functions
    protected: 
        //! Maintain the delSets and delOther data structure for particle i
        //! If it returns false there was a problem and a global re-triangulation is triggered.
        bool getDelSets(int i);
        //!resize all neighMax-related arrays
        void resetLists();
        //!sort points along a Hilbert curve for data locality
        void spatialSorting();


    //protected member variables
    protected:
        //! A flag that determines whether the GPU RNG is the same every time.
        //!The default is for randomness, but maintain the option for testing.
        //!Must be known upon initialization!
        bool Reproducible;

        //!A flag to notify whether cells of different type have added tension terms at their interface
        bool useTension;
        //!A flag that notifies the existence of any particle exclusions (for which the net force is set to zero by fictitious external forces)
        bool particleExclusions;

        //!The value of inter-cell surface tension to apply to cells of different type
        Dscalar gamma;

        //!An array of displacements used only for the CPU-only branch of operating
        GPUArray<Dscalar2> displacements;

        //!delSet.data[n_idx(nn,i)] are the previous and next consecutive delaunay neighbors,
        //!orientationally ordered, of point i (for use in computing forces on GPU)
        GPUArray<int2> delSets;
        //delOther.data[n_idx(nn,i)] contains the index of the "other" delaunay neighbor. i.e., the mutual
        //!neighbor of delSet.data[n_idx(nn,i)].y and delSet.data[n_idx(nn,i)].z that isn't point i
        GPUArray<int> delOther;

        //!In GPU mode, interactions are computed "per voronoi vertex"...forceSets are summed up to get total force on a particle
        GPUArray<Dscalar2> forceSets;

    //public member variables
    public:
        //! Read from a database what the time of the simulation when saved was
        Dscalar SimTime;
        //! Count the number of times "performTimeStep" has been called
        int Timestep;
        //! Determines how frequently he spatial sorter be called...once per sortPeriod Timesteps. When sortPeriod < 0 no sorting occurs
        int sortPeriod;
        //!A flag that determins if a spatial sorting is due to occur this Timestep
        bool spatialSortThisStep;

        //!an array of net forces on cels
        GPUArray<Dscalar2> forces;

        //!"exclusions" zero out the force on a cell...the external force needed to do this is stored in external_forces
        GPUArray<Dscalar2> external_forces;
        //!An array containing the indices of excluded particles
        GPUArray<int> exclusions;


        //!Some function-timing-related scalars
        Dscalar triangletiming, forcetiming;
    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabase;
    };

#endif
