#ifndef STRUCTURES_H
#define STRUCTURES_H

/////////////////
//This header file defines useful structures for computing triangulations...
//... points as a pair of coordinates, edges as a pair of vertex numbers, triangles as a triplet
//... etc.
/////////////////

//a few function protocols needed below...definitions are in functions.h
//since some structures need to be able to call this function...

#include "std_include.h"

bool CircumCircle(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2, Dscalar x3, Dscalar y3,Dscalar &xc, Dscalar &yc, Dscalar &r);
inline Dscalar TriangleArea(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2);

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


struct DelaunayCell
    {
    //a class that has the number of delaunay neighbors, and their positions relative to the vertex
    //can compute the voronoi cell's area and perimeter
    public:
        int n; //number of delaunay neighbors
        std::vector< Dscalar2 > Dneighs;
        std::vector<std::pair <Dscalar,int> > CWorder;
        std::vector< Dscalar2> Vpoints;
        Dscalar Varea;
        Dscalar Vperimeter;
        bool Voro; //have the voronoi points of the cell already been calculated?

        DelaunayCell(){Voro=false;};

        void setSize(int nn){n=nn;Dneighs.resize(n);Voro=false;};

        //find CW order of neighbors
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

        //find the positions of the voronoi cell around the vertex
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

struct edge
    {//contains a pair of integers {i,j} of vertex labels
    public:
        int i;
        int j;
        edge(){};
        ~edge(){};
        edge(int ii, int jj){i=ii;j=jj;};
    };

struct triangle
    {//contains a triplet of integers {i,j,k} of vertex labels
    public:
        int i;
        int j;
        int k;
        triangle(){};
        ~triangle(){};
        triangle(int ii, int jj,int kk){i=ii;j=jj;k=kk;};
    };

struct triangulation
    {//information to specific a triangulation of verticies,
     //has vectors of edges and triangles
     //contains a triplet of integers {i,j,k} of vertex labels
     //getNeighbors will sift through the triangles to look for neighbors of vertex i
    public:
        int nTriangles;
        int nEdges;
        std::vector<edge> edges;
        std::vector<triangle> triangles;

        //searches for vertices that are in triangles with vertex i.
        //Returns a sorted list (according to integer value, NOT CW order!), with no duplicates
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
