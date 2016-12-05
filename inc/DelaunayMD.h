//DelaunayMD.h
#ifndef DELAUNAYMD_H
#define DELAUNAYMD_H

using namespace std;
#include "std_include.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"
#include "indexer.h"
#include "HilbertSort.h"

#include "DelaunayLoc.h"
#include "DelaunayCGAL.h"

class DelaunayMD
    {
    protected:

        std::vector<pt> pts;          //vector of points to triangulate
        int N;                       //number of vertices
        bool triangulated;            //has a triangulation been performed?

        Dscalar cellsize;
        cellListGPU celllist;


        //neighbor lists
        GPUArray<int> neigh_num;
        GPUArray<int> neighs;

        //an array that holds (particle, neighbor_number) info to avoid intra-warp divergence in force calculation?
        GPUArray<int2> NeighIdxs;
        int NeighIdxNum;

        Index2D n_idx;
        //circumcenter lists
        GPUArray<int3> circumcenters;

        int neighMax;
        int NumCircumCenters;

        int Fails;
        int FullFails;

        //repair is a vector of zeros (everything is fine) and ones (that index needs to be repaired)
        GPUArray<int> repair;
        vector<int> NeedsFixing;

        bool GPUcompute;
        int timestep;
        bool neighMaxChange;

    public:
        gpubox Box;
        box CPUbox;
        DelaunayLoc delLoc;
        GPUArray<Dscalar2> points;      //vector of particle positions
        Dscalar polytiming,ringcandtiming,reducedtiming,tritiming,tritesttiming,geotiming,totaltiming;
        Dscalar gputiming,cputiming;
        Dscalar repPerFrame;
        int skippedFrames;
        int GlobalFixes;

        //maps between particle index and spatially sorted tag
        vector<int> idxToTag;
        vector<int> tagToIdx;

        void getPoints(GPUArray<Dscalar2> &ps){ps = points;};
        //constructors
        DelaunayMD(){triangulated=false;cellsize=2.0;};

        //initialization functions
        void initialize(int n);
        void randomizePositions(Dscalar boxx, Dscalar boxy);

        //move particles
        void movePoints(GPUArray<Dscalar2> &displacements);
        void movePointsCPU(GPUArray<Dscalar2> &displacements);

        //utility functions
        void resetDelLocPoints();
        void spatiallySortPoints();
        void updateCellList();
        void reportCellList();
        void reportPos(int i);
        void touchPoints(){ArrayHandle<Dscalar2> h(points,access_location::host,access_mode::readwrite);};
        void updateNeighIdxs();

        //only use the CPU:
        void setCPU(){GPUcompute = false;};

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
