#ifndef STRUCTURES_H
#define STRUCTURES_H

//
//This header file defines useful structures for computing triangulations...
//... points as a pair of coordinates, edges as a pair of vertex numbers, triangles as a triplet

#include "std_include.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//a few function protocols needed below...definitions are in functions.h
//since some structures need to be able to call this function...
bool CircumCircle(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2, Dscalar x3, Dscalar y3,Dscalar &xc, Dscalar &yc, Dscalar &r);
inline Dscalar TriangleArea(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2);

/*!
 * Really the Voronoi cell of a Delaunay vertex. Given the relative positions of the vertices 
 * Delaunay neighbors, this puts the neighbors in clockwise order and calculates the Voronoi vertices
 * of the Voronoi cell. Also calculates the area and perimeter of the Voronoi cell.
 */
struct DelaunayCell
    {
    public:
        int n; //!<number of delaunay neighbors
        std::vector< Dscalar2 > Dneighs; //!< The relative positions of the Delaunay neighbors
        std::vector<std::pair <Dscalar,int> > CWorder; //!< A structure to put the neighbors in oriented order
        std::vector< Dscalar2> Vpoints; //!< The voronoi vertices
        Dscalar Varea; //!< The area of the cell
        Dscalar Vperimeter; //!< The perimeter of the cell
        bool Voro; //!<have the voronoi points of the cell already been calculated?

        DelaunayCell(){Voro=false;};

        //! Declare how many neighbors the cell has
        void setSize(int nn){n=nn;Dneighs.resize(n);Voro=false;};

        //!find CW order of neighbors
        void getCW()
            {
            CWorder.resize(n);
            Vpoints.resize(n);
            for (int ii = 0; ii < n; ++ii)
                {
                CWorder[ii].first=atan2(Dneighs[ii].y,Dneighs[ii].x);
                CWorder[ii].second=ii;
                };
            sort(CWorder.begin(),CWorder.begin()+n);
            }

        //!Find the positions of the voronoi cell around the vertex
        void getVoro()
            {
            //first, put the points in clockwise order
            getCW();

            //calculate the voronoi points as the circumcenter of the origin,p_i,p_{i+1}
            Dscalar2 ori;
            ori.x=0.0;ori.y=0.0;
            for (int ii=0; ii < n; ++ii)
                {
                Dscalar xc,yc,rad;
                bool placeholder; // Circumcircle is a function with a type
                Dscalar2 p1 = Dneighs[CWorder[ii].second];
                Dscalar2 p2 = Dneighs[CWorder[((ii+1)%n)].second];
                placeholder = CircumCircle(ori.x,ori.y,p1.x,p1.y,p2.x,p2.y,xc,yc,rad);
                Vpoints[ii]=make_Dscalar2(xc,yc);
                };

            Voro=true;
            };

        //!Calculate the area and perimeter of the voronoi cell.
        void Calculate()
            {
            if (!Voro) getVoro();
            Varea = 0.0;
            Vperimeter = 0.0;
            for (int ii = 0; ii < n; ++ii)
                {
                Dscalar2 p1 = Vpoints[ii];
                Dscalar2 p2 = Vpoints[((ii+1)%n)];
                Varea += TriangleArea(p1.x,p1.y,p2.x,p2.y);
                Dscalar dx = p1.x-p2.x;
                Dscalar dy = p1.y-p2.y;
                Vperimeter += sqrt(dx*dx+dy*dy);
                };
            };

    };

/// contains a pair of integers of vertex labels
struct edge
    {
    public:
        int i;
        int j;
        edge(){};
        ~edge(){};
        edge(int ii, int jj){i=ii;j=jj;};
    };

//contains a triplet of integers {i,j,k} of vertex labels
struct triangle
    {
    public:
        int i;
        int j;
        int k;
        triangle(){};
        ~triangle(){};
        triangle(int ii, int jj,int kk){i=ii;j=jj;k=kk;};
    };

/*!
 * Contains the information needed to specify a triangulation of vertices.
 * It has vectors of edges and triangles, and getNeighbors will sift through the
 * triangles to look for neighbors of vertex i
 */
struct triangulation
    {
    public:
        int nTriangles; //!<The number of triangles
        int nEdges; //!< the number of edges
        std::vector<edge> edges; //!< a vector of edges
        std::vector<triangle> triangles; //!<a vector of triangles

        //!searches for vertices that are in triangles with vertex i. Returns a sorted list (according to integer value, NOT CW order!), with no duplicates.
        void getNeighbors(int i, std::vector<int> &neighs)
            {
            neighs.clear();
            for (int tt = 0; tt < nTriangles; ++tt)
                {
                if (i == triangles[tt].i){neighs.push_back(triangles[tt].j);neighs.push_back(triangles[tt].k);};
                if (i == triangles[tt].j){neighs.push_back(triangles[tt].k);neighs.push_back(triangles[tt].i);};
                if (i == triangles[tt].k){neighs.push_back(triangles[tt].i);neighs.push_back(triangles[tt].j);};
                };
            sort(neighs.begin(),neighs.end());
            neighs.erase( unique( neighs.begin(),neighs.end() ), neighs.end() );
            };

    };


#undef HOSTDEVICE

#endif
