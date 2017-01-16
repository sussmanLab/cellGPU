#ifndef DELAUNAYMD_H
#define DELAUNAYMD_H

#include "std_include.h"
#include "Simple2DActiveCell.h"
#include "Simple2DActiveCell.cuh"
#include "cellListGPU.cuh"
#include "cellListGPU.h"
#include "DelaunayLoc.h"
#include "DelaunayCGAL.h"
#include "DelaunayMD.cuh"
#include "HilbertSort.h"


/*!
 * DelaunayMD is a core engine class, capable of taking a set of points
 * in a periodic domain, performing Delaunay triangulations on them, testing whether
 * those triangulations are valid on either the CPU or GPU, and locally repair
 * invalid triangulations on the CPU.
 */
//! Perform and test triangulations in an MD setting, using kernels in \ref DelaunayMDKernels
class DelaunayMD : public Simple2DActiveCell
    {
    //public functions first
    public:
        //!The constructor!
        DelaunayMD();
        //!A default initialization scheme
        void initializeDelMD(int n);
        /*!
        \param global defaults to true.
        When global is set to true, the CPU branch will try the local repair scheme.
        This is generally slower, but if working in a regime where things change
        very infrequently, it may be faster.
        */
        //!Enforce CPU-only operation.
        void setCPU(bool global = true){GPUcompute = false;globalOnly=global;};
        //!write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        //!read positions from text file...for debugging
        void readTriangulation(ifstream &infile);

        //!A very hacky and wrong soft-sphere repulsion between neighbors... strictly for testing purposes
        void repel(GPUArray<Dscalar2> &disp,Dscalar eps);

        //!move particles on the GPU
        void movePoints(GPUArray<Dscalar2> &displacements);
        //!move particles on the CPU
        void movePointsCPU(GPUArray<Dscalar2> &displacements);
        //!Transfer particle data from the GPU to the CPU for use by delLoc
        void resetDelLocPoints();
        //!Perform a spatial sorting of the particles to try to maintain data locality
        void spatiallySortPoints();

        //!Update the cell list structure after particles have moves
        void updateCellList();
        //!update the NieghIdxs data structure
        void updateNeighIdxs();

    //protected functions
    protected:
        //!construct the global periodic triangulation point-by-point using non-CGAL methods
        void fullTriangulation();
        //!Globally construct the triangulation via CGAL
        void globalTriangulationCGAL(bool verbose = false);

        //!build the auxiliary data structure containing the indices of the particle circumcenters from the neighbor list
        void getCircumcenterIndices(bool secondtime=false,bool verbose = false);

        //!Test the current neighbor list to see if it is still a valid triangulation. GPU function
        void testTriangulation();
        //!Test the validity of the triangulation on the CPU
        void testTriangulationCPU();
        //!repair any problems with the triangulation on the CPU
        void repairTriangulation(vector<int> &fixlist);
        //!A workhorse function that calls the appropriate topology testing and repairing routines
        void testAndRepairTriangulation(bool verb = false);

    //public member variables
    public:
        //!The class' local Delaunay tester/updater
        DelaunayLoc delLoc;

        //!Collect statistics of how many triangulation repairs are done per frame, etc.
        Dscalar repPerFrame;
        //!How often were all circumcenters empty (so that no data transfers and no repairs were necessary)?
        int skippedFrames;
        //!How often were global re-triangulations performed?
        int GlobalFixes;

        
    protected:
        cellListGPU celllist;        //!<The associated cell list structure
        Dscalar cellsize;            //!<The size of the cell list's underlying grid

        //!A 2dIndexer for computing where in the GPUArray to look for a given particles neighbors
        Index2D n_idx;
        //!An upper bound for the maximum number of neighbors that any cell has
        int neighMax;

        //!An array that holds (particle, neighbor_number) info to avoid intra-warp divergence in GPU
        //!-based force calculations that might be used by child classes
        GPUArray<int2> NeighIdxs;
        //!A utility integer to help with NeighIdxs
        int NeighIdxNum;

        //!A data structure that holds the indices of particles forming the circumcircles of the Delaunay Triangulation
        GPUArray<int3> circumcenters;
        //!The number of circumcircles...for a periodic system, this should never change. This provides one check that local updates to the triangulation are globally consistent
        int NumCircumCenters;

        /*! 
        \todo the current implementation of anyCircumcenterTestFailed and
        completeRetriangulationPerformed has a malloc cost on every time step. Revise this.
        */
        //!A flag that can be accessed by child classes... serves as notification that any change in the network topology has occured
        int anyCircumcenterTestFailed;
        //!A flag that notifies that a global re-triangulation has been performed
        int completeRetriangulationPerformed;
        //!A flag that notifies that the maximum number of neighbors may have changed, necessitating resizing of some data arrays
        bool neighMaxChange;

        //!A a vector of zeros (everything is fine) and ones (that index needs to be repaired)
        GPUArray<int> repair;
        //!A smaller vector that, after testing the triangulation, contains the particle indices that need their local topology to be updated.
        vector<int> NeedsFixing;

        /*!When running on the CPU, should only global retriangulations be performed,
        or should local test-and-updates still be performed? Depending on parameters
        simulated, performance here can be quite difference, since the circumcircle test
        itself is CPU expensive
        */
        //!When true, the CPU branch will execute global retriangulations through CGAL on every time step
        bool globalOnly;

        //!Count the number of times that testAndRepair has been called, separately from the derived class' time
        int timestep;

    };



#endif
