//DelaunayTri.h
#ifndef DELAUNAYTRI_H
#define DELAUNAYTRI_H

using namespace std;
extern "C" {
#include "triangle.h"

};
//A class that computes periodic Delaunay triangulations by calling Shewchuk's Triangle program on a 9X copied version of the original cell
class DelaunayTri
    {

    private:
        std::vector<float> pts;          //vector of points to triangulate
        int nV;                       //number of vertices
        bool sorted;                  //are the points sorted
        bool triangulated;            //has a triangulation been performed?
    
    public:
        DelaunayTri();
        //constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayTri(std::vector<float> points){setPoints(points);};

        void setPoints(std::vector<float> points);

        //default call... update this whenever a better algorithm is implemented
        void getTriangulation();//name change required since "triangulate" is triangle's algorithm

        //simple unit test
        void testDel(int numpts,int tmax);
    };



#endif
