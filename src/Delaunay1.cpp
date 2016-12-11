#define EPSILON 1e-16

#include <cmath>
#include <algorithm>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/time.h>

using namespace std;

#include "Delaunay1.h"


void DelaunayNP::setPoints(vector<Dscalar> points)
    {
    nV=points.size()/2;
    sorted=false;
    triangulated = false;
    sortmap.clear();
    sortmap.reserve(nV);
    mapi.clear();
    mapi.resize(nV);
    pts.resize(nV);
    for (unsigned int ii = 0; ii<nV; ++ii)
        {
        Dscalar2 point;
        point.x=points[ii*2];point.y=points[ii*2+1];
        pts[ii]=point;
        pair<Dscalar2, int> pp;
        pp.first = point;
        pp.second=ii;
        sortmap.push_back(pp);
        };
    };

void DelaunayNP::setPoints(vector<Dscalar2> points)
    {
    nV=points.size();
    pts=points;
    sorted=false;
    triangulated = false;
    sortmap.clear();
    sortmap.reserve(nV+3);
    mapi.clear();
    mapi.resize(nV);
    for (unsigned int ii = 0; ii<nV; ++ii)
        {
        pair<Dscalar2, int> pp;
        pp.first.x=pts[ii].x;
        pp.first.y=pts[ii].y;
        pp.second=ii;
        sortmap.push_back(pp);
        };
    };

void DelaunayNP::sortPoints()
    {
    sort(sortmap.begin(),sortmap.end());
    for (int nn = 0; nn < nV; ++nn)
        mapi[sortmap[nn].second]=nn;
    sorted = true;
    };

void DelaunayNP::triangulate()
    {//update if a better DT algorithm is implemented
    naiveBowyerWatson();
    };

void DelaunayNP::naiveBowyerWatson()
    {
    bool incircle;
    Dscalar xmin,xmax,ymin,ymax,dx,dy,dmax,xcen,ycen;
    int ntri;
    Dscalar2 xtest,c;
    Dscalar rad;
    if (!sorted) sortPoints();
    edge nullEdge(-1,-1);
    int emax   = 3*(nV+1); //really 3n-3-k for n points with k points in the convex hull
    int trimax = 2*(nV+1); //really 2n-2-k for '' ''   ''
    vector<bool> complete(trimax,false);

    DT.edges.resize(emax);
    DT.triangles.resize(trimax);
    DT.nEdges=0;

    //First, calculate a bounding supertriangle
    xmin = sortmap[0].first.x;
    ymin = sortmap[0].first.y;
    xmax=sortmap[nV-1].first.x;
    ymax=ymin;
    Dscalar xx, yy;
    for(int ii=0;ii < nV;++ii)
        {
        xx=sortmap[ii].first.x;
        yy=sortmap[ii].first.y;
        if (yy < ymin) ymin = yy;
        if (yy > ymax) ymax = yy;
        };
    dx=xmax-xmin;
    dy=ymax-ymin;
    dmax=3.0*dy;
    if (dx > dy) dmax = 3.0*dx;
    xcen = 0.5*(xmax+xmin);
    ycen = 0.5*(ymax+ymin);

    Dscalar2 point;
    pair<Dscalar2, int> newvert;

    point.x = xcen-0.866*dmax;
    point.y = ycen-0.5*dmax;
    newvert.first=point;
    newvert.second=nV;
    sortmap.push_back(newvert);
    point.x = xcen+0.866*dmax;
    point.y = ycen-0.5*dmax;
    newvert.first=point;
    newvert.second=nV+1;
    sortmap.push_back(newvert);
    point.x = xcen;
    point.y = ycen+dmax;
    newvert.first=point;
    newvert.second=nV+2;
    sortmap.push_back(newvert);
    triangle Tri(nV,nV+1,nV+2);
    DT.triangles[0]=Tri;

    ntri = 1;

    //begin loop to insert each vertex into the triangulation
    Dscalar xp, yp;
    for(int ii = 0; ii < nV; ++ii)
        {
        xp = sortmap[ii].first.x;
        yp = sortmap[ii].first.y;
        Dscalar2 Xp;
        Xp.x=xp;Xp.y=yp;
        DT.nEdges=0;

        //loop over constructed triangles, find the ones where (xp,yp) is in the circumcenter
        for (int jj = 0; jj < ntri; ++jj)
            {
            if(!complete[jj])
                {
                triangle tr(DT.triangles[jj].i,DT.triangles[jj].j,DT.triangles[jj].k);
                incircle = Circumcircle(Xp,  sortmap[tr.i].first, sortmap[tr.j].first,  sortmap[tr.k].first,c,rad);
                if (c.x+rad < xp) complete[jj]=true;
                if (incircle)
                    {
                    edge E1(DT.triangles[jj].i,DT.triangles[jj].j);
                    edge E2(DT.triangles[jj].j,DT.triangles[jj].k);
                    edge E3(DT.triangles[jj].k,DT.triangles[jj].i);
                    DT.edges[DT.nEdges+0]=E1;
                    DT.edges[DT.nEdges+1]=E2;
                    DT.edges[DT.nEdges+2]=E3;
                    DT.nEdges =DT.nEdges+3;
                    DT.triangles[jj]=DT.triangles[ntri-1];
                    complete[jj]=complete[ntri-1];
                    ntri--;
                    jj--;
                    };

                };
            };//end loop over triangles

        //loop over edges, tagging any that are repeats
        for (int jj = 0; jj < DT.nEdges-1; ++jj)
            {
            for (int kk = jj+1; kk < DT.nEdges;++kk)
                {
                if((DT.edges[jj].i==DT.edges[kk].j)&&(DT.edges[jj].j==DT.edges[kk].i))
                    {
                    DT.edges[jj]=nullEdge;
                    DT.edges[kk]=nullEdge;
                    };
                };
            };

        //arrange new edges for the current point (clockwise order, so that the above repeat-checking loop is correct)
        for (int jj = 0; jj < DT.nEdges;++jj)
            {
            if((DT.edges[jj].i >= 0 ) && (DT.edges[jj].j>=0) )
                {
                triangle newtri(DT.edges[jj].i,DT.edges[jj].j,ii);
                DT.triangles[ntri]=newtri;
                complete[ntri] = false;
                ntri++;
                };
            };

        };//end loop over vertex insertion

    //finally, prune the triangles to get rid of any that contain the bounding super-triangle
    for (int ii = 0; ii < ntri; ++ii)
        {
        if( (DT.triangles[ii].i >= nV)||(DT.triangles[ii].j >= nV)||(DT.triangles[ii].k >= nV) )
            {
            DT.triangles[ii]=DT.triangles[ntri-1];
            ntri--;
            ii--;
            };
        };

    DT.nTriangles=ntri;

    triangulated = true;

    };

void DelaunayNP::printTriangulation(int maxprint)
    {
    if (!triangulated)
        {
        cout << "No triangulation has been performed -- nothing to print" <<endl;
        return;
        };
    for (int ii = 0; ii < min(maxprint,nV); ++ii)
        {
        cout <<"{"<<sortmap[ii].first.x<<","<<sortmap[ii].first.y<<"},";
        };
    cout << endl;
    for (int tt = 0; tt < min(maxprint,DT.nTriangles); ++tt)
        cout <<"{"<<DT.triangles[tt].i << ",  " <<DT.triangles[tt].j << ",  " <<DT.triangles[tt].k <<"},";
    cout << endl;
    };

void DelaunayNP::writeTriangulation(ofstream &outfile)
    {
    outfile << nV <<"\t"<<DT.nTriangles<<"\t"<<DT.nEdges<<endl;
    for (int ii = 0; ii < nV ; ++ii)
        outfile << sortmap[ii].first.x <<"\t" <<sortmap[ii].first.y <<endl;
    for (int ii = 0; ii < DT.nTriangles; ++ii)
        outfile << DT.triangles[ii].i <<"\t" <<DT.triangles[ii].j<<"\t"<<DT.triangles[ii].k<<endl;
    for (int ii = 0; ii < DT.nEdges; ++ii)
        outfile << DT.edges[ii].i <<"\t" <<DT.edges[ii].j<<endl;
    };

void DelaunayNP::testDel(int numpts, int tmax,bool verbose)
    {
    cout << "Timing the base, non-periodic routine..." << endl;
    nV = numpts;
    Dscalar boxa = sqrt(numpts)+1.0;
    vector<Dscalar> ps2(2*numpts);
    Dscalar maxx = 0.0;
    int randmax = 1000000;
    for (int i=0;i<numpts;++i)
        {
        Dscalar x =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        Dscalar y =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        ps2[i*2]=x;
        ps2[i*2+1]=y;
        };

    clock_t tstart,tstop;
    tstart = clock();

    for (int ii = 0; ii < tmax; ++ii)
        {
        setPoints(ps2);
//        sorted=false;
        triangulate();
        };

    tstop=clock();
    Dscalar timing = (tstop-tstart)/(Dscalar)CLOCKS_PER_SEC/(Dscalar)tmax;
    cout << "average time per complete triangulation = " << timing<< endl;

    };


