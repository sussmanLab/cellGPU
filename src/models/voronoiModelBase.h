#ifndef voronoiModelBase_H
#define voronoiModelBase_H

#include "Simple2DActiveCell.h"
#include "cellListGPU.cuh"
#include "cellListGPU.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "structures.h"
#include "voronoiModelBase.cuh"


/*! \file voronoiModelBase.h */
//! Perform and test triangulations in an MD setting, using kernels in \ref voronoiModelBaseKernels
/*!
 * voronoiModelBase is a core engine class, capable of taking a set of points
 * in a periodic domain, performing Delaunay triangulations on them, testing whether
 * those triangulations are valid on either the CPU or GPU, and locally repair
 * invalid triangulations on the CPU.

 * Voronoi models have their topology taken care of by the underlying triangulation, and so child 
   classes just need to implement an energy functions (and corresponding force law)
 */
class voronoiModelBase : public Simple2DActiveCell
    {
    public:
        //!The constructor!
        voronoiModelBase();
        //!A default initialization scheme
        void initializeVoronoiModelBase(int n, bool useGPU = true);
        void reinitialize(int neighborGuess);
        //!Enforce CPU-only operation.
        /*!
        \param global defaults to true.
        When global is set to true, the CPU branch will try the local repair scheme.
        This is generally slower, but if working in a regime where things change
        very infrequently, it may be faster.
        */
        void setCPU(bool global = true){GPUcompute = false;globalOnly=global;delGPU.setGPUcompute(false);};
        virtual void setGPU(){GPUcompute = true; delGPU.setGPUcompute(true);};
        //!write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        //!read positions from text file...for debugging
        void readTriangulation(ifstream &infile);

        //! change the box dimensions, and rescale the positions of particles
        virtual void setRectangularUnitCell(double Lx, double Ly);

        //!update/enforce the topology
        virtual void enforceTopology();

        //!Declare which particles are to be excluded (exes[i]!=0)
        void setExclusions(vector<int> &exes);
        //!set a new simulation box, update the positions of cells based on virtual positions, and then recompute the geometry
        void alterBox(PeriodicBoxPtr _box);

        //virtual functions that need to be implemented
        //!In voronoi models the number of degrees of freedom is the number of cells
        virtual int getNumberOfDegreesOfFreedom(){return Ncells;};

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<double2> & displacements,double scale = 1.);
        //!return the forces
        virtual void getForces(GPUArray<double2> &forces){forces = cellForces;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<double2> & returnForces(){return cellForces;};

        //!Compute cell geometry on the CPU
        virtual void computeGeometryCPU();
        //!call gpu_compute_geometry kernel caller
        virtual void computeGeometryGPU();

        //!allow for cell division, according to a vector of model-dependent parameters
        virtual void cellDivision(const vector<int> &parameters,const vector<double> &dParams);

        //!Kill the indexed cell by simply removing it from the simulation
        virtual void cellDeath(int cellIndex);

        //!move particles on the GPU
        void movePoints(GPUArray<double2> &displacements,double scale);
        //!move particles on the CPU
        void movePointsCPU(GPUArray<double2> &displacements,double scale);

        //!Update the cell list structure after particles have moved
        void updateCellList();
        //!update the NieghIdxs data structure
        void updateNeighIdxs();
        //set number of threads
        virtual void setOmpThreads(int _number){ompThreadNum = _number;delGPU.setOmpThreads(_number);};

    //protected functions
    protected:
        //!sort points along a Hilbert curve for data locality
        void spatialSorting();

        //!Globally construct the triangulation via CGAL
        void globalTriangulationCGAL(bool verbose = false);
        //!Globally construct the triangulation via DelGPU
        void globalTriangulationDelGPU(bool verbose = false);

        //!repair any problems with the triangulation on the CPU
        void repairTriangulation(vector<int> &fixlist);
        //! after a CGAL triangulation, need to populate delGPU's voroCur structure in order for compute geometry to work
        void populateVoroCur();
        //! call getDelSets for all particles
        void allDelSets();

        //! Maintain the delSets and delOther data structure for particle i
        bool getDelSets(int i);
        //!resize all neighMax-related arrays
        void resetLists();
        //!do resize and resetting operations common to cellDivision and cellDeath
        void resizeAndReset();

        //Some functions associated with derivates of voronoi vertex positions or cell geometries
        //!The derivative of a voronoi vertex position with respect to change in the first cells position
        Matrix2x2 dHdri(double2 ri, double2 rj, double2 rk);
        //!Derivative of the area of cell i with respect to the position of cell j
        double2 dAidrj(int i, int j);
        //!Derivative of the perimeter of cell i with respect to the position of cell j
        double2 dPidrj(int i, int j);
        //!Second derivative of area w/r/t voronoi and cell position
        Matrix2x2 d2Areadvdr(Matrix2x2 &dvpdr, Matrix2x2 &dvmdr);
        //!Second derivative of perimeter w/r/t voronoi and cell position
        Matrix2x2 d2Peridvdr(Matrix2x2 &dvdr, Matrix2x2 &dvmdr, Matrix2x2 &dvpdr,double2 vm, double2 v, double2 vp);
        //!second derivatives of voronoi vertex with respect to cell positions
        vector<double> d2Hdridrj(double2 rj, double2 rk, int jj);

    //public member variables
    public:
        //!The class' local Delaunay tester/updater
        DelaunayGPU delGPU;

        //!Collect statistics of how many triangulation repairs are done per frame, etc.
        double repPerFrame;
        //!How often were all circumcenters empty (so that no data transfers and no repairs were necessary)?
        int skippedFrames;
        //!How often were global re-triangulations performed?
        int GlobalFixes;
        //!"exclusions" zero out the force on a cell...the external force needed to do this is stored in external_forces
        GPUArray<double2> external_forces;
        //!An array containing the indices of excluded particles
        GPUArray<int> exclusions;

    protected:
        //!The size of the cell list's underlying grid
        double cellsize;            
        //!An upper bound for the maximum number of neighbors that any cell has
        int neighMax;

        //!An array that holds (particle, neighbor_number) info to avoid intra-warp divergence in GPU
        //!-based force calculations that might be used by child classes
        GPUArray<int2> NeighIdxs;
        //!A utility integer to help with NeighIdxs
        int NeighIdxNum;

        //!A flag that can be accessed by child classes... serves as notification that any change in the network topology has occured
        GPUArray<int> anyCircumcenterTestFailed;
        //!A flag that notifies that a global re-triangulation has been performed
        int completeRetriangulationPerformed;
        //!A flag that notifies that the maximum number of neighbors may have changed, necessitating resizing of some data arrays
        bool neighMaxChange;

        //!A a vector of zeros (everything is fine) and ones (that index needs to be repaired)...also used as an assist array in updateNeighIdxs
        GPUArray<int> repair;
        //!A smaller vector that, after testing the triangulation, contains the particle indices that need their local topology to be updated.
        vector<int> NeedsFixing;

        //!When true, the CPU branch will execute global retriangulations through CGAL on every time step
        /*!When running on the CPU, should only global retriangulations be performed,
        or should local test-and-updates still be performed? Depending on parameters
        simulated, performance here can be quite difference, since the circumcircle test
        itself is CPU expensive
        */
        bool globalOnly;
        //!Count the number of times that testAndRepair has been called, separately from the derived class' time
        int timestep;
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
        GPUArray<double2> forceSets;
    };

#endif
