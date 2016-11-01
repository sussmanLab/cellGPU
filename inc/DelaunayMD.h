//DelaunayMD.h
#ifndef DELAUNAYMD_H
#define DELAUNAYMD_H

using namespace std;
//#include "functions.h"
//#include "structures.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

#include "DelaunayLoc.h"

class DelaunayMD
    {
    private:
        GPUArray<float2> points;      //vector of particle positions

        std::vector<pt> pts;          //vector of points to triangulate
        int N;                       //number of vertices
        bool triangulated;            //has a triangulation been performed?

        float cellsize;
        cellListGPU celllist;
        gpubox Box;
        
        DelaunayLoc delLoc;

        GPUArray<int> neigh_num; 
        GPUArray<int> neighs;
        int NeighMax;

    public:
        float polytiming,ringcandtiming,reducedtiming,tritiming,tritesttiming,geotiming,totaltiming;


        //constructors
        DelaunayMD(){triangulated=false;cellsize=2.0;};
        //constructor via a vector of point objects
//        DelaunayMD(std::vector<pt> &points, box &bx){setPoints(points);setBox(bx);};
        //constructor via a vector of scalars, {x1,y1,x2,y2,...}
//        DelaunayMD(std::vector<float> &points,box &bx){setPoints(points);setBox(bx);};

        //initialization functions
        void initialize(int n);
        void randomizePositions(float boxx, float boxy);

        //utility functions
        void resetDelLocPoints();
        void updateCellList();
        void reportCellList();

        //construct complete triangulation
        void fullTriangulation();




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
