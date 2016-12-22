//DelaunayMD.h
#ifndef DELAUNAYMD_H
#define DELAUNAYMD_H

#include "std_include.h"

#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.cuh"
#include "gpucell.h"
#include "indexer.h"
#include "HilbertSort.h"

#include "DelaunayLoc.h"
#include "DelaunayCGAL.h"

class DelaunayMD
    {
    protected:
        int N;                       //number of vertices
        cellListGPU celllist;        //the cell list structure
        Dscalar cellsize;            //size of its grid

        //neighbor lists and their indexer. And the maximum number of neighbors any particle has
        GPUArray<int> neigh_num;
        GPUArray<int> neighs;
        Index2D n_idx;
        int neighMax;

        //an array that holds (particle, neighbor_number) info to avoid intra-warp divergence in force calculation?
        GPUArray<int2> NeighIdxs;
        int NeighIdxNum;

        //circumcenter lists
        GPUArray<int3> circumcenters;
        int NumCircumCenters;

        //flags that can be accessed by child classes...has any change in the network topology occured? Has a complete retriangulation been performed (necessitating changes in array sizes)?
        int Fails;
        int FullFails;
        //has the maximum neighbor number changed?
        bool neighMaxChange;

        //repair is a vector of zeros (everything is fine) and ones (that index needs to be repaired)
        GPUArray<int> repair;
        vector<int> NeedsFixing;

        //flags...should the GPU be used? If no, what CPU routine should be run (global vs. local retriangulations)?
        bool globalOnly;

        //this class' time        
        int timestep;
        std::vector<Dscalar2> pts;          //vector of points to triangulate...for delLoc purposes


    public:
        GPUArray<Dscalar2> points;      //vector of particle positions
        //the GPU and CPU boxes owned by this object
        gpubox Box;
        bool GPUcompute;

        //the local Delaunay tester/updater
        DelaunayLoc delLoc;

        //statistics of how many triangulation repairs are done per frame, etc.
        Dscalar repPerFrame;
        int skippedFrames;
        int GlobalFixes;

        //maps between particle index and spatially sorted tag...together with itt and tti (the versions of idxToTag and tagToIdx stored from the last spatial sorting) enables one to keep track of initial particle indices based on current index/tag combinations
        vector<int> itt;
        vector<int> tti;
        vector<int> idxToTag;
        vector<int> tagToIdx;

        /////
        //Member functions
        /////

        //constructors
        DelaunayMD(){cellsize=2.0;};

        //initialization functions
        void initializeDelMD(int n);
        void randomizePositions(Dscalar boxx, Dscalar boxy);

        //move particles
        void movePoints(GPUArray<Dscalar2> &displacements);
        void movePointsCPU(GPUArray<Dscalar2> &displacements);

        //utility functions
        void resetDelLocPoints();
        void spatiallySortPoints();

        //why use templates when you can type more?
        void reIndexArray(GPUArray<int> &array);
        void reIndexArray(GPUArray<Dscalar> &array);
        void reIndexArray(GPUArray<Dscalar2> &array);

        void updateCellList();
        void updateNeighIdxs();

        void getPoints(GPUArray<Dscalar2> &ps){ps = points;};
        
        //only use the CPU... pass global = false to not call CGAL to test/fix the triangulation
        void setCPU(bool global = true){GPUcompute = false;globalOnly=global;};

        //construct complete triangulation point-by-point
        void fullTriangulation();
        //resort to a method that globally constructs the triangulation
        void globalTriangulationCGAL(bool verbose = false);

        //construct circumcenters structure from neighbor list
        void getCircumcenterIndices(bool secondtime=false,bool verbose = false);

        //Test the current neigh list to see if it is still a valid triangulation
        //If it isn't, fix it on the cpu
        void testTriangulation();
        void testTriangulationCPU(); //force CPU-based computation
        void repairTriangulation(vector<int> &fixlist);
        void testAndRepairTriangulation(bool verb = false);

        //write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        //read positions from text file
        void readTriangulation(ifstream &infile);


        //soft-sphere repulsion....for testing
        void repel(GPUArray<Dscalar2> &disp,Dscalar eps);


    };



#endif
