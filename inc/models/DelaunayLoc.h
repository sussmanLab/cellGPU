#ifndef DELAUNAYLOC_H
#define DELAUNAYLOC_H

#include "Delaunay1.h"
#include "gpuarray.h"
#include "gpubox.h"
#include "cellListGPU.h"
using namespace std;

/*! \file DelaunayLoc.h */
 //!A CPU-based class for locally constructing the Delaunay triangulation of part of a point set
/*!
 *
 * One of the workhorse engines of the hybrid scheme, DelaunayLoc provides methods for finding
 * the Delaunay neighbors of a given point by doing only local calculations. This is done by
 * calculating the candidate 1-ring of vertex i -- a set of points for which the set of Delaunay
 * neighbors of i is a strict subset -- by calculating points contained in the circumcircles
 * of a set of points that form a bounding polygon around i. This can be used to completely construct
 * a DT of the entire point set in the periodic domain (by calling either DelaunayNP or DelaunayCGAL
 * to go from the candidate 1-ring to the actual list of neighbors), or to locally repair a part of a
 * triangulation that has become non-Delaunay.
 * This function operates strictly on the CPU
 */
class DelaunayLoc
    {
    public:
        DelaunayLoc(){triangulated=false;cellsize=2.0;Box = make_shared<gpubox>();};
        //!constructor via a vector of Dscalar2 objects
        DelaunayLoc(std::vector<Dscalar2> &points, gpubox &bx){setPoints(points);setBox(bx);};
        //!constructor via a vector of scalars, {x1,y1,x2,y2,...}
        DelaunayLoc(std::vector<Dscalar> &points,gpubox &bx){setPoints(points);setBox(bx);};

        void setPoints(ArrayHandle<Dscalar2> &points, int N); //!<Set points by passing an ArrayHandle
        void setPoints(GPUArray<Dscalar2> &points); //!<Set points via a GPUarray of Dscalar2's
        void setPoints(std::vector<Dscalar2> &points); //!<Set points via a vector of Dscalar2's
        void setPoints(std::vector<Dscalar> &points);   //!<Set the points via a vector of Dscalar's
        void setBox(gpubox &bx);                        //!<Set the box
        void setBox(BoxPtr bx){Box=bx;};                        //!<Set the box
        void setCellSize(Dscalar cs){cellsize=cs;};     //!<Set the cell size of the underlying grid

        void initialize(Dscalar csize);                 //!<Initialize various things, based on a given cell size for the underlying grid

        //!Find the indices of an enclosing polygon of vertex i
        void getPolygon(int i, vector<int> &P0,vector<Dscalar2> &P1);
        //!Find a candidate set of possible points in the 1-ring of vertex i
        void getOneRingCandidate(int i, vector<int> &DTringIdx,vector<Dscalar2> &DTring);
        //!If the candidate 1-ring is large, try to reduce it before triangulating the whole thing
        void reduceOneRing(int i, vector<int> &DTringIdx,vector<Dscalar2> &DTring);
        //!Collect some statistics about the functioning of the oneRing algorithms
        int cellschecked,candidates; //statistics for the above function

        //!Return the neighbors of vertex i, sorted in CW order, along with a voronoi cell with calculated geometric properties
        void triangulatePoint(int i, vector<int> &neighbors, DelaunayCell &DCell,bool timing=false);
        //!Just get the neighbors of vertex i, sorted in clockwise order. Calculated by the DelaunayNP class
        void getNeighbors(int i, vector<int> &neighbors);
        //!Just get the neighbors of vertex i, sorted in clockwise order. Calculated by the DelaunayCGAL class
        bool getNeighborsCGAL(int i, vector<int> &neighbors);

        //!Test whether the passed list of neighbors are the Delaunay neighbors of vertex i
        bool testPointTriangulation(int i, vector<int> &neighbors, bool timing=false);
        //!Given a vector of circumcircle indices, label particles that are part of non-empty circumcircles
        void testTriangulation(vector< int > &ccs, vector<bool> &points, bool timing=false);
        //!return the gpubox
        virtual gpubox & returnBox(){return *(Box);};

        //!A public variable that stores the triangulation as sets of (i,j,k) vertices when this class is used to generate the entire triangulation of the periodic point set.
        triangulation DT;
        //!output part of triangulation to screen for debugging purposes
        void printTriangulation(int maxprint);
        //!write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        //!print the location of a point to the screen...for debugging.
        void printPoint(int i){cout <<pts[i].x << " " <<pts[i].y << endl;};

        //!simple testing function
        void testDel(int numpts,int tmax,double err, bool verbose);

        //!Various aids for timing functions
        Dscalar polytiming,ringcandtiming,reducedtiming,tritiming,tritesttiming,geotiming,totaltiming;

    protected:
        std::vector<Dscalar2> pts;    //!<vector of points to triangulate
        int nV;                       //!<number of vertices
        bool triangulated;            //!<has a triangulation been performed?

        Dscalar cellsize;               //!<Sets how fine a grid to use in the cell list
        BoxPtr Box;             //!< A box to calculate relative distances in a periodic domain.

        vector<int> DTringIdxCGAL; //!<A vector of Delaunay neighbor indicies that can be repeatedly re-written
        vector<Dscalar2> DTringCGAL;//!<A vector of Delaunay neighbors that can be repeatedly re-written
        //!A cell list for speeding up the calculation of the candidate 1-ring
        cellListGPU cList;
    };
#endif
