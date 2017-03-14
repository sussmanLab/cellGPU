#ifndef DELAUNAY1_H
#define DELAUNAY1_H

#include "std_include.h"
#include "functions.h"
#include "structures.h"
using namespace std;

/*! \file Delaunay1.h */
 //! construct Delaunay Triangulation of a 2D, non-periodic point set via Bowyer-Watson algorithm
/*!
DelaunayNP implements a naive Bowyer-watson algorithm to compute the Delaunay Triangulation
of a non-periodic set of points. Ideally this method should not be called (in favor of CGAL-
based triangulations), but if for some reason CGAL is not available this can be used in a pinch.
In particular, it will be slower and more prone to crashes, as CGAL has a robust error catching
scheme.
 */
class DelaunayNP
    {
    public:
        DelaunayNP(){sorted=false;triangulated=false;};
        //!constructor via a vector of point objects
        DelaunayNP(std::vector<Dscalar2> points){setPoints(points);};
        //!constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayNP(std::vector<Dscalar> points){setPoints(points);};

        void setSorted(bool s){sorted=s;};
        //!Set the points you want to triangulate
        void setPoints(std::vector<Dscalar2> points);
        void setPoints(std::vector<Dscalar> points);

        //!A debugging function, prints the location of a point
        void printPoint(int i){cout <<pts[i].x << " " <<pts[i].y << endl;};
        //!A debugging function... gets the sorted point i
        void getSortedPoint(int i,Dscalar2 &point)
            {
            if (i >= nV) {cout << "invalid sort point access" << endl;
            cout << i << "   " << nV << endl;
            cout.flush();};
            point = sortmap[i].first;
            };

        //!Figure out the original point that the sorted set of points refers to
        int deSortPoint(int i){return sortmap[i].second;};

        //!sorts points by their x-coordinate...necessary for the algorithm
        void sortPoints();

        std::vector<int> mapi; //!<a map to sorted coords...mapi[j] tells you the position in sortmap that vertex j appears in

        //!default call... update this whenever a better algorithm is implemented
        void triangulate();

        //!calculate the Delaunay triangulation via an O(nV^1.5) version ofBowyer-Watson
        void naiveBowyerWatson();

        //!a public variable that stores the triangulation as sets of (i,j,k) vertices, referring to the order of sortmap entries
        triangulation DT;
        //!output part of triangulation to screen
        void printTriangulation(int maxprint);
        //!write triangulation to text file
        void writeTriangulation(ofstream &outfile);

        //!simple unit test
        void testDel(int numpts,int tmax,bool verbose);

    private:
        std::vector<Dscalar2> pts;      //!<vector of points to triangulate
        int nV;                       //!<number of vertices
        bool sorted;                  //!<are the points sorted
        bool triangulated;            //!<has a triangulation been performed?
        std::vector< pair<Dscalar2, int> > sortmap;    //!<map from sorted points back to input points
    };

#endif
