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
 CPU and GPU implementation of the Delaunay triangulation
 *It makes use of a locallity lemma described in (doi: 10.1109/ISVD.2012.9).
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

        //!Given a point set, fill the int arrays with a Delaunay triangulation
        void globalDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!Given a point set and a putative triangulation of it, check the validity and replace input triangulation with correct one
        void testAndRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!initialization function
        void initialize(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx);
        //!function call to change the maximum number of neighbors per point
        void resize(const int nmax);
        //! update the size of the cell list bins 
        void setCellListSize(double csize);
        //!Only update the cell list
        void updateList(GPUArray<double2> &points);
        //!Set the box from a 
        void setBox(PeriodicBoxPtr bx){Box=bx;};
        //!Set the safetyMode flag...If safetyMode is false and the assumptions are not satisfied, program will be wrong with (possibly) no warning!
        void setSafetyMode(bool _sm){safetyMode=_sm;};
        //!set a flag to use GPU routines. When false, use CPU routines
        void setGPUcompute(bool flag){GPUcompute=flag;};
        //!Set the number of threads to ask openMP to use during CPU-based triangulation loops
    	void setOMPthreads(unsigned int num){OMPThreadsNum=num;};
        //!A lightweight profiler to use when timing the functionality of this class
        multiProfiler prof;
        //! A box to calculate relative distances in a periodic domain.
        PeriodicBoxPtr Box;
        //! The maximum number of neighbors any point has
        int MaxSize;
        //!A helper array containing the positions of voronoi vertices associated with every 1-ring
        GPUArray<double2> GPUVoroCur;

    protected:
        //!Given point set, test the quality of a triangulation on the GPU
        void testTriangulation(GPUArray<double2> &points);
        //!Given point set, test the quality of a triangulation on the CPU
        void testTriangulationCPU(GPUArray<double2> &points);
        //!build the auxiliary data structure containing the indices of the particle circumcircles from the neighbor list on the CPU
        void getCircumcirclesCPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        //!build the auxiliary data structure on the GPU
        void getCircumcirclesGPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!In the global triangulation  branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the GPU
        void Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        //!In the global triangulation  branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the CPU
        void Voronoi_Calc_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        //!In the global triangulation branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the GPU
        bool get_neighbors(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);
        //!In the global triangulation branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the CPU
        bool get_neighbors_CPU(GPUArray<double2> &points,GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum);

        //!In the testAndRepair branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the GPU
        void voronoiCalcRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!In the testAndRepair branch of the code, calculate a candidate one-ring by finding an enclosing polygon of points on the CPU
        void voronoiCalcRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //! In the testAndRepair branch of the code, turn candidate 1-rings into actual 1-rings on the GPU
        bool computeTriangulationRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //! In the testAndRepair branch of the code, turn candidate 1-rings into actual 1-rings on the CPU
        bool computeTriangulationRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList);
        //!prep the cell list for accelerating local point searches
        void initializeCellList();

        //!If false, the user is guaranteeing that the current maximum number of neighbors per point will not be exceeded
        bool safetyMode = false;
        //!A flag to notify whether the cellList structure has been updated...used during testAndRepair
        bool cListUpdated=false;

        //All of these arrays should be accessed by the GPU_idx!

        //!Repair the parts of the triangulation associated with the given repairList
        void locallyRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList, int numberToRepair=-1);

        //!A helper array containing the positions of the delaunay positions associated with every 1-ring of neighboring points
        GPUArray<double2> GPUDelNeighsPos;
        //!A helper array containing the size of circumcircles associated with point(kidx) and GPUDelNeighsPos(kidx,i) and GPUDelNeighsPos(kidx,i+1)
        GPUArray<double> GPUVoroCurRad;
        //!A helper array containing the indices of the points forming the 1-ring of each point
        GPUArray<int> GPUPointIndx;
        //!A helper array for the testAndRepair branch containing indices of points forming circumcircles
        GPUArray<int3> delGPUcircumcircles;
        //!A helper array used to keep track of points to repair in the testAndRepair branch of operation
        GPUArray<int>repair;
        //!An array that holds a single int keeping track of maximum 1-ring size
        GPUArray<int> maxOneRingSize;

        //!A int containing the number of points
        int Ncells;
        //!The number of circumcircles. Due to current assumptions in the code, this is always 2*Ncells
        int NumCircumcircles;

        //!A utility structure, used to compute circumcenter sets on the GPU
        GPUArray<int> circumcirclesAssist;
        //!A flag that tells the code to use either CPU or GPU routines
        bool GPUcompute;
        //!Variable that keeps the number of threads used by OpenMP
        unsigned int OMPThreadsNum=1;

        //!A 2dIndexer for computing where in the GPUArray to look for a given particles neighbors GPU
        Index2D GPU_idx;

        //!A cell list for speeding up the calculation of the candidate 1-ring
        cellListGPU cList;
        //!keep track of the linear size of the cells used by the cellListGPU object
        double cellsize;
    };
#endif
