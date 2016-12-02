//DelaunayLoc.h
#ifndef DELAUNAYLOC_H
#define DELAUNAYLOC_H

using namespace std;
#include "std_include.h"
#include "box.h"
#include "cell.h"
#include "Delaunay1.h"

class DelaunayLoc
    {
    private:
        std::vector<pt> pts;          //vector of points to triangulate
        int nV;                       //number of vertices
        bool triangulated;            //has a triangulation been performed?

        Dscalar cellsize;
        grid clist;
        box Box;


    public:
        Dscalar polytiming,ringcandtiming,reducedtiming,tritiming,tritesttiming,geotiming,totaltiming;


        DelaunayLoc(){triangulated=false;cellsize=2.0;};
        //constructor via a vector of point objects
        DelaunayLoc(std::vector<pt> &points, box &bx){setPoints(points);setBox(bx);};
        //constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayLoc(std::vector<Dscalar> &points,box &bx){setPoints(points);setBox(bx);};

        void setPoints(std::vector<pt> &points);
        void setPoints(std::vector<Dscalar> &points);
        void setBox(box &bx);
        void setCellSize(Dscalar cs){cellsize=cs;};

        void initialize(Dscalar csize);

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
        //"neighbors" returns a vector of the index of Delaunay neighbors of vertex i, sorted in clockwise order
        void getNeighbors(int i, vector<int> &neighbors);
        //"neighbors" returns a vector of the index of Delaunay neighbors of vertex i, sorted in clockwise order...calculated via CGAL
        void getNeighborsCGAL(int i, vector<int> &neighbors);


        //test triangulations... "neighbors" must be in CW or CCW order!
        bool testPointTriangulation(int i, vector<int> &neighbors, bool timing=false);
        //void testTriangulation(vector< vector<int> > &neighbors, vector<bool> &points, bool timing=false);
        void testTriangulation(vector< int > &neighbors, vector<bool> &points, bool timing=false);


        //a public variable (for now) that stores the triangulation as sets of (i,j,k) vertices
        triangulation DT;

        //output part of triangulation to screen
        void printTriangulation(int maxprint);
        //write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        void printPoint(int i){cout <<pts[i].x << " " <<pts[i].y << endl;};

        //simple test
        void testDel(int numpts,int tmax,double err, bool verbose);

    };

#endif
