//DelaunayMD.h
#ifndef DELAUNAYMD_H
#define DELAUNAYMD_H

using namespace std;
//#include "functions.h"
//#include "structures.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"
#include "indexer.h"

#include "DelaunayLoc.h"
#include "DelaunayCGAL.h"

class DelaunayMD
    {
    protected:

        std::vector<pt> pts;          //vector of points to triangulate
        int N;                       //number of vertices
        bool triangulated;            //has a triangulation been performed?

        float cellsize;
        cellListGPU celllist;


        //neighbor lists
        GPUArray<int> neigh_num; 
        GPUArray<int> neighs;
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

    public:
        gpubox Box;
        box CPUbox;
        DelaunayLoc delLoc;
        GPUArray<float2> points;      //vector of particle positions
        float polytiming,ringcandtiming,reducedtiming,tritiming,tritesttiming,geotiming,totaltiming;
        float gputiming,cputiming;
        float repPerFrame;
        int skippedFrames;
        int GlobalFixes;

        void getPoints(GPUArray<float2> &ps){ps = points;};
        //constructors
        DelaunayMD(){triangulated=false;cellsize=2.0;};

        //initialization functions
        void initialize(int n);
        void randomizePositions(float boxx, float boxy);

        //move particles
        void movePoints(GPUArray<float2> &displacements);

        //utility functions
        void resetDelLocPoints();
        void updateCellList();
        void reportCellList();
        void reportPos(int i);
        void touchPoints(){ArrayHandle<float2> h(points,access_location::host,access_mode::readwrite);};

        //only use the CPU:
        void setCPU(){GPUcompute = false;};

        //construct complete triangulation point-by-point
        void fullTriangulation();
        //resort to a method that globally constructs the triangulation
        void globalTriangulation(bool verbose = false);
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
        void repel(GPUArray<float2> &disp,float eps);

        //old functions
        void setPoints(std::vector<pt> &points);
        void setPoints(std::vector<float> &points);
        void setBox(gpubox &bx);
        void setCellSize(float cs){cellsize=cs;};


        //find indices of enclosing polygon of vertex i (helper function for the next function)
        void getPolygon(int i, vector<int> &P0,vector<pt> &P1);
        //finds a candidate set of possible points in the 1-ring of vertex i
        void getOneRingCandidate(int i, vector<int> &DTringIdx,vector<pt> &DTring);
        //checks if the one ring can be reduced by changing the initial polygon
        void reduceOneRing(int i, vector<int> &DTringIdx,vector<pt> &DTring);
        int cellschecked,candidates; //statistics for the above function

        //default call... update this whenever a better algorithm is implemented
        //"neighbors" returns a vector of the index of Delaunay neighbors of vertex i, sorted in clockwise order
        void triangulatePoint(int i, vector<int> &neighbors, DelaunayCell &DCell,bool timing=false);


        //a public variable (for now) that stores the triangulation as sets of (i,j,k) vertices
        triangulation DT;

        //output part of triangulation to screen
        void printTriangulation(int maxprint);
        //write triangulation to text file


    };



#endif
