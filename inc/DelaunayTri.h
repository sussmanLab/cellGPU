//DelaunayTri.h
#ifndef DELAUNAYTRI_H
#define DELAUNAYTRI_H

using namespace std;
using namespace voroguppy;
extern "C" {
#include "triangle.h"
#include "box.h"
#include "structures.h"

};
//A class that computes periodic Delaunay triangulations by calling Shewchuk's Triangle program on a 9X copied version of the original cell
class DelaunayTri
    {

    private:
        std::vector<float> pts;          //vector of points to triangulate
        int nV;                       //number of vertices
        bool sorted;                  //are the points sorted
        bool triangulated;            //has a triangulation been performed?
        box Box;
    
    public:
        DelaunayTri();
        //constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayTri(std::vector<float> points){setPoints(points);};

        void setPoints(std::vector<float> &points);
        void setBox(box &bx);

        //default call... update this whenever a better algorithm is implemented
        void getTriangulation();//name change required since "triangulate" is triangle's algorithm
        void getTriangulation9();//name change required since "triangulate" is triangle's algorithm

        void getNeighbors(vector<float> &points,int idx, vector<int> &neighs);
        void getNeighbors(vector<pt> &points,int idx, vector<int> &neighs);

        //simple unit test
        void testDel(int numpts,int tmax,bool verbose);
    };



#endif
