//Delaunay1.h
#ifndef DELAUNAY1_H
#define DELAUNAY1_H

using namespace std;
#include "functions.h"
#include "structures.h"

namespace voroguppy
{

class DelaunayNP
    {

    private:
        std::vector<pt> pts;          //vector of points to triangulate
        int nV;                       //number of vertices
        bool sorted;                  //are the points sorted
        bool triangulated;            //has a triangulation been performed?
        std::vector< pair<pt, int> > sortmap;    //map from sorted points back to input points

    public:
        DelaunayNP(){sorted=false;triangulated=false;};
        //constructor via a vector of point objects
        DelaunayNP(std::vector<pt> points){setPoints(points);};
        //constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayNP(std::vector<float> points){setPoints(points);};

        void setSorted(bool s){sorted=s;};
        void setPoints(std::vector<pt> points);
        void setPoints(std::vector<float> points);

        void printPoint(int i){cout <<pts[i].x << " " <<pts[i].y << endl;};
        void getSortedPoint(int i,pt &point)
            {
            if (i >= nV) {cout << "invalid sort point access" << endl;
            cout << i << "   " << nV << endl;
            cout.flush();};
            point = sortmap[i].first;
            };
        int deSortPoint(int i){return sortmap[i].second;};

        //sorts points by their x-coordinate
        void sortPoints();

        std::vector<int> mapi; //a map to sorted coords...mapi[j] tells you the position in sortmap that vertex j appears in

        //default call... update this whenever a better algorithm is implemented
        void triangulate();

        //calculate the Delaunay triangulation via an O(nV^1.5) version ofBowyer-Watson
        void naiveBowyerWatson();

        //a public variable (for now) that stores the triangulation as sets of (i,j,k) vertices, referring to the order of sortmap entries
        triangulation DT;

        //output part of triangulation to screen
        void printTriangulation(int maxprint);
        //write triangulation to text file
        void writeTriangulation(ofstream &outfile);

        //simple unit test
        void testDel(int numpts,int tmax);
    };


};
#endif
