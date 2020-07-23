#ifndef DELAUNAYGPU_H
#define DELAUNAYGPU_H

#include "gpuarray.h"
#include "periodicBoundaries.h"
#include "cellListGPU.h"
#include "multiProfiler.h"

using namespace std;

/*! \file DelaunayGPU.h */
 //!A GPU-based class for locally constructing the Delaunay triangulation of part of a point set
/*!
 *GPU implementation of the DT.
 *It makes use of a locallity lema described in (doi: 10.1109/ISVD.2012.9).
 *It will only make the repair of the topology in case it is necessary.
 *Steps are detailed as in paper.
 */

class DelaunayGPU
    {
	public:

		//!blank constructor
		DelaunayGPU();
        //!Constructor + initialiation
        DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx);
		//!Destructor
		~DelaunayGPU(){};

        //!initialization function
        void initialize(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx);

        //!function call to change the maximum number of neighbors per point
        void resize(const int nmax);

        //!Initialize various things, based on a given cell size for the underlying grid
        void setList(double csize, GPUArray<double2> &points);
        //!Only update the cell list
        void updateList(GPUArray<double2> &points);
        //!Set the box
        void setBox(periodicBoundaries &bx);
        void setBox(PeriodicBoxPtr bx){Box=bx;};
        //!Set the cell size of the underlying grid
        void setCellSize(double cs){cellsize=cs;};

        //!build the auxiliary data structure containing the indices of the particle circumcircles from the neighbor list
        void getCircumcirclesCPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //go through GPU routines
        void setGPUcompute(bool flag)
        {
            GPUcompute=flag;
        };

        //!Given a point set, fill the int arrays with a Delaunay triangulation
        void globalDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!Given a point set and a putative triangulation of it, check the validity and replace input triangulation with correct one
        void testAndRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!Repair the parts of the triangulation associated with the given repairList
        void locallyRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList, int numberToRepair=-1);


        multiProfiler prof;

        //!Set the safetyMode flag...IF safetyMode is false and the assumptions are not satisfied, program will be wrong with (possibly) no warning
        void setSafetyMode(bool _sm){safetyMode=_sm;};

        //!< A box to calculate relative distances in a periodic domain.
        PeriodicBoxPtr Box;

        bool cListUpdated;
        //! The maximum number of neighbors any point has
        int MaxSize;

    private:
        //Functions used by the GPU DT
        void testTriangulation(GPUArray<double2> &points);
        void testTriangulationCPU(GPUArray<double2> &points);
        //!build the auxiliary data structure on the GPU
        void getCircumcirclesGPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!Main function of this class, it performs the Delaunay triangulation
        void Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        bool get_neighbors(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        void Voronoi_Calc_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        bool get_neighbors_CPU(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!testing an alternate memory pattern for local repairs
        void voronoiCalcRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        void voronoiCalcRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!same memory pattern, for getNeighbors
        bool computeTriangulationRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        bool computeTriangulationRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!prep the cell list
        void initializeCellList();

        //!If false, the user is guaranteeing that the current maximum number of neighbors per point will not be exceeded
        bool safetyMode = false;

    protected:

        //!A helper array used for the triangulation on the GPU, before the topology is known
        GPUArray<double2> GPUVoroCur;
        GPUArray<double2> GPUDelNeighsPos;
        GPUArray<double> GPUVoroCurRad;
        GPUArray<int> GPUPointIndx;

        GPUArray<int> neighs;
        GPUArray<double2> pts;
        GPUArray<int3> delGPUcircumcircles;
        GPUArray<int>repair;

        //!An array that holds a single int keeping track of maximum 1-ring size
        GPUArray<int> maxOneRingSize;

        int Ncells;
        int NumCircumcircles;

        //!A utility list -- currently used to compute circumcenter sets on the GPU
        GPUArray<int> sizeFixlist;
        int size_fixlist;
        //flag that tells the code to use either CPU or GPU routines
        bool GPUcompute;

        //!A 2dIndexer for computing where in the GPUArray to look for a given particles neighbors GPU
        Index2D GPU_idx;

        //!A cell list for speeding up the calculation of the candidate 1-ring
        cellListGPU cList;
        double cellsize;
    };
#endif
