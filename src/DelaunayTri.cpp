using namespace std;
#define ANSI_DECLARATIONS
#define dbl float
#define REAL double
#define EPSILON 1e-12
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

#include "box.h"
#include "DelaunayTri.h"

extern "C" void triangulate(char*, triangulateio*,triangulateio*,triangulateio*);

DelaunayTri::DelaunayTri()
    {

    };

void DelaunayTri::setPoints(std::vector<float> &points)
    {
    pts=points;
    };

void DelaunayTri::setBox(voroguppy::box &bx)
    {
    dbl b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    Box.setGeneral(b11,b12,b21,b22);
    };

void DelaunayTri::getNeighbors(vector<pt> &points,int idx, vector<int> &neighs)
    {
    struct triangulateio in, mid, out, vorout;
    neighs.clear();

    /* Define input points. */
    int numpts = points.size();
    in.numberofpoints = numpts;
    in.numberofpointattributes = 1;
    in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
    for (int ii = 0; ii < points.size(); ++ii)
        {
        in.pointlist[2*ii]=points[ii].x;
        in.pointlist[2*ii+1]=points[ii].y;
        };

    in.pointattributelist = (REAL *) malloc(in.numberofpoints *
            in.numberofpointattributes *
            sizeof(REAL));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointattributelist[ii]= 0.0;

    in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointmarkerlist[ii]= 0.0;

    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 1;
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));

//    in.regionlist[0] = 0.5;
//    in.regionlist[1] = 5.0;
//    in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
//    in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

    mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of point attributes is zero: */
    mid.pointattributelist = (REAL *) NULL;
    mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
    mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    mid.triangleattributelist = (REAL *) NULL;
    mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
    /* Needed only if segments are output (-p or -c) and -P not used: */
    mid.segmentlist = (int *) NULL;
    /* Needed only if segments are output (-p or -c) and -P and -B not used: */
    mid.segmentmarkerlist = (int *) NULL;
    mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
    mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

    vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
    /* Needed only if -v switch used and number of attributes is not zero: */
    vorout.pointattributelist = (REAL *) NULL;
    vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
    vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

    /* Triangulate the points.  Switches are chosen to read and write a  */
    /*   PSLG (p), preserve the convex hull (c), number everything from  */
    /*   zero (z), assign a regional attribute to each element (A), and  */
    /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
    /*   neighbor list (n).                                              */

//    triangulate("pczAevnQ", &in, &mid, &vorout);
//
//  Here, get edges, quietly, and number from zero. incremental?
    triangulate("zeQ", &in, &mid, &vorout);

//    printf("%i points\n",mid.numberofpoints);
//    printf("finding edges of point %i\n",idx);

    for (int ee =0; ee < mid.numberofedges; ++ee)
        {
        int i1, i2;
        i1 = mid.edgelist[2*ee];
        i2 = mid.edgelist[ee*2+1];
//        printf("%4d  %4d   %4d\n ", mid.neighborlist[ee*3 ],mid.neighborlist[2*ee+1],mid.neighborlist[3*ee+2]);
        if ( i1==idx)
            neighs.push_back(i2);
        if ( i2==idx)
            neighs.push_back(i1);
        };


//    printf("Initial triangulation:\n\n");
//    report(&mid, 1, 1, 1, 1, 1, 0);
//    printf("Initial Voronoi diagram:\n\n");
//    report(&vorout, 0, 0, 0, 0, 1, 1);

    /* Attach area constraints to the triangles in preparation for */
    /*   refining the triangulation.                               */

    /* Needed only if -r and -a switches used: */
    mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
    mid.trianglearealist[0] = 3.0;
    mid.trianglearealist[1] = 1.0;

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `out'.                                    */

    out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of attributes is zero: */
    out.pointattributelist = (REAL *) NULL;
    out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    out.triangleattributelist = (REAL *) NULL;

    /* Refine the triangulation according to the attached */
    /*   triangle area constraints.                       */

//    triangulate("pranzBP", &mid, &out, (struct triangulateio *) NULL);


    free(in.pointlist);
    free(in.pointattributelist);
    free(in.pointmarkerlist);
    free(in.regionlist);
    free(mid.pointlist);
    free(mid.pointattributelist);
    free(mid.pointmarkerlist);
    free(mid.trianglelist);
    free(mid.triangleattributelist);
    free(mid.trianglearealist);
    free(mid.neighborlist);
    free(mid.segmentlist);
    free(mid.segmentmarkerlist);
    free(mid.edgelist);
    free(mid.edgemarkerlist);
    free(vorout.pointlist);
    free(vorout.pointattributelist);
    free(vorout.edgelist);
    free(vorout.normlist);
    free(out.pointlist);
    free(out.pointattributelist);
    free(out.trianglelist);
    free(out.triangleattributelist);

    };


void DelaunayTri::getNeighbors(vector<float> &points,int idx, vector<int> &neighs)
    {
    struct triangulateio in, mid, out, vorout;

    /* Define input points. */
    int numpts = points.size()/2;
    in.numberofpoints = numpts;
    in.numberofpointattributes = 1;
    in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
    for (int ii = 0; ii < points.size(); ++ii)
        {
        in.pointlist[ii]=points[ii];
        };

    in.pointattributelist = (REAL *) malloc(in.numberofpoints *
            in.numberofpointattributes *
            sizeof(REAL));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointattributelist[ii]= 0.0;

    in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointmarkerlist[ii]= 0.0;

    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 1;
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));

//    in.regionlist[0] = 0.5;
//    in.regionlist[1] = 5.0;
//    in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
//    in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

    mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of point attributes is zero: */
    mid.pointattributelist = (REAL *) NULL;
    mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
    mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    mid.triangleattributelist = (REAL *) NULL;
    mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
    /* Needed only if segments are output (-p or -c) and -P not used: */
    mid.segmentlist = (int *) NULL;
    /* Needed only if segments are output (-p or -c) and -P and -B not used: */
    mid.segmentmarkerlist = (int *) NULL;
    mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
    mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

    vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
    /* Needed only if -v switch used and number of attributes is not zero: */
    vorout.pointattributelist = (REAL *) NULL;
    vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
    vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

    /* Triangulate the points.  Switches are chosen to read and write a  */
    /*   PSLG (p), preserve the convex hull (c), number everything from  */
    /*   zero (z), assign a regional attribute to each element (A), and  */
    /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
    /*   neighbor list (n).                                              */

//    triangulate("pczAevnQ", &in, &mid, &vorout);
//
//  Here, get edges, quietly, and number from zero
    triangulate("zeQ", &in, &mid, &vorout);

    printf("%i points\n",mid.numberofpoints);
    printf("finding edges of point %i\n",idx);

    for (int ee =0; ee < mid.numberofedges; ++ee)
        {
        int i1, i2;
        i1 = mid.edgelist[2*ee];
        i2 = mid.edgelist[ee*2+1];
//        printf("%4d  %4d   %4d\n ", mid.neighborlist[ee*3 ],mid.neighborlist[2*ee+1],mid.neighborlist[3*ee+2]);
        if ( i1==idx)
            printf("%i \t",i2);
        if ( i2==idx)
            printf("%i \t",i1);
        };
    printf("\n");


//    printf("Initial triangulation:\n\n");
//    report(&mid, 1, 1, 1, 1, 1, 0);
//    printf("Initial Voronoi diagram:\n\n");
//    report(&vorout, 0, 0, 0, 0, 1, 1);

    /* Attach area constraints to the triangles in preparation for */
    /*   refining the triangulation.                               */

    /* Needed only if -r and -a switches used: */
    mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
    mid.trianglearealist[0] = 3.0;
    mid.trianglearealist[1] = 1.0;

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `out'.                                    */

    out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of attributes is zero: */
    out.pointattributelist = (REAL *) NULL;
    out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    out.triangleattributelist = (REAL *) NULL;

    /* Refine the triangulation according to the attached */
    /*   triangle area constraints.                       */

//    triangulate("pranzBP", &mid, &out, (struct triangulateio *) NULL);


    free(in.pointlist);
    free(in.pointattributelist);
    free(in.pointmarkerlist);
    free(in.regionlist);
    free(mid.pointlist);
    free(mid.pointattributelist);
    free(mid.pointmarkerlist);
    free(mid.trianglelist);
    free(mid.triangleattributelist);
    free(mid.trianglearealist);
    free(mid.neighborlist);
    free(mid.segmentlist);
    free(mid.segmentmarkerlist);
    free(mid.edgelist);
    free(mid.edgemarkerlist);
    free(vorout.pointlist);
    free(vorout.pointattributelist);
    free(vorout.edgelist);
    free(vorout.normlist);
    free(out.pointlist);
    free(out.pointattributelist);
    free(out.trianglelist);
    free(out.triangleattributelist);

    };



void DelaunayTri::getTriangulation()
    {
    cout.flush();
    struct triangulateio in, mid, out, vorout;

    /* Define input points. */
    int np = pts.size()/2;
    int numpts = pts.size()/2;
    in.numberofpoints = numpts;
    in.numberofpointattributes = 1;
    in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
    for (int ii = 0; ii < pts.size(); ++ii)
        {
        if (ii % 2 ==0)
            in.pointlist[ii]=pts[ii];
        if (ii % 2 ==1)
            in.pointlist[ii]=pts[ii];
        };

    in.pointattributelist = (REAL *) malloc(in.numberofpoints *
            in.numberofpointattributes *
            sizeof(REAL));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointattributelist[ii]= 0.0;

    in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointmarkerlist[ii]= 0.0;

    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 1;
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));

//    in.regionlist[0] = 0.5;
//    in.regionlist[1] = 5.0;
//    in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
//    in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

    mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of point attributes is zero: */
    mid.pointattributelist = (REAL *) NULL;
    mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
    mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    mid.triangleattributelist = (REAL *) NULL;
    mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
    /* Needed only if segments are output (-p or -c) and -P not used: */
    mid.segmentlist = (int *) NULL;
    /* Needed only if segments are output (-p or -c) and -P and -B not used: */
    mid.segmentmarkerlist = (int *) NULL;
    mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
    mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

    vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
    /* Needed only if -v switch used and number of attributes is not zero: */
    vorout.pointattributelist = (REAL *) NULL;
    vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
    vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

    /* Triangulate the points.  Switches are chosen to read and write a  */
    /*   PSLG (p), preserve the convex hull (c), number everything from  */
    /*   zero (z), assign a regional attribute to each element (A), and  */
    /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
    /*   neighbor list (n).                                              */

//    triangulate("pczAevnQ", &in, &mid, &vorout);
//
//  Here, get edges, quietly, and number from zero
    triangulate("zeQ", &in, &mid, &vorout);
/*
    printf("%i points\n",mid.numberofpoints);
    printf("finding edges of point 31\n");

    for (int ee =0; ee < mid.numberofedges; ++ee)
        {
        int i1, i2;
        i1 = mid.edgelist[2*ee];
        i2 = mid.edgelist[ee*2+1];
//        printf("%4d  %4d   %4d\n ", mid.neighborlist[ee*3 ],mid.neighborlist[2*ee+1],mid.neighborlist[3*ee+2]);
        if ( i1==31)
            printf("%i \t",i2);
        if ( i2==31)
            printf("%i \t",i1);
        };
    printf("\n");
*/

//    printf("Initial triangulation:\n\n");
//    report(&mid, 1, 1, 1, 1, 1, 0);
//    printf("Initial Voronoi diagram:\n\n");
//    report(&vorout, 0, 0, 0, 0, 1, 1);

    /* Attach area constraints to the triangles in preparation for */
    /*   refining the triangulation.                               */

    /* Needed only if -r and -a switches used: */
    mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
    mid.trianglearealist[0] = 3.0;
    mid.trianglearealist[1] = 1.0;

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `out'.                                    */

    out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of attributes is zero: */
    out.pointattributelist = (REAL *) NULL;
    out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    out.triangleattributelist = (REAL *) NULL;

    /* Refine the triangulation according to the attached */
    /*   triangle area constraints.                       */

//    triangulate("pranzBP", &mid, &out, (struct triangulateio *) NULL);


    free(in.pointlist);
    free(in.pointattributelist);
    free(in.pointmarkerlist);
    free(in.regionlist);
    free(mid.pointlist);
    free(mid.pointattributelist);
    free(mid.pointmarkerlist);
    free(mid.trianglelist);
    free(mid.triangleattributelist);
    free(mid.trianglearealist);
    free(mid.neighborlist);
    free(mid.segmentlist);
    free(mid.segmentmarkerlist);
    free(mid.edgelist);
    free(mid.edgemarkerlist);
    free(vorout.pointlist);
    free(vorout.pointattributelist);
    free(vorout.edgelist);
    free(vorout.normlist);
    free(out.pointlist);
    free(out.pointattributelist);
    free(out.trianglelist);
    free(out.triangleattributelist);

    };

void DelaunayTri::getTriangulation9()
    {
    cout.flush();
    struct triangulateio in, mid, out, vorout;

    /* Define input points. */
    dbl b11,b12,b21,b22;
    Box.getBoxDims(b11,b12,b21,b22);
    int np = pts.size()/2;
    int numpts = 9*pts.size()/2;
    in.numberofpoints = numpts;
    in.numberofpointattributes = 1;
    in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
    for (int xx = -1; xx <= 1; ++xx)
        {
        for (int yy = -1; yy <= 1;++yy)
            {
            for (int ii = 0; ii < pts.size(); ++ii)
                {
                int idx = ((yy+1)+3*(xx+1))*np*2+ii;
                if (ii % 2 ==0)
                    in.pointlist[idx]=pts[ii]+xx*b11;
                if (ii % 2 ==1)
                    in.pointlist[idx]=pts[ii]+yy*b22;
                };
            }
        };

    in.pointattributelist = (REAL *) malloc(in.numberofpoints *
            in.numberofpointattributes *
            sizeof(REAL));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointattributelist[ii]= 0.0;

    in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
    for (int ii = 0; ii < numpts; ++ii)
        in.pointmarkerlist[ii]= 0.0;

    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 1;
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));

    in.regionlist[0] = 0.5;
    in.regionlist[1] = 5.0;
    in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
    in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */
    /*
    in.numberofpoints = 4;
    in.numberofpointattributes = 1;
    in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
    in.pointlist[0] = 0.0;
    in.pointlist[1] = 0.0;
    in.pointlist[2] = 1.0;
    in.pointlist[3] = 0.0;
    in.pointlist[4] = 1.0;
    in.pointlist[5] = 10.0;
    in.pointlist[6] = 0.0;
    in.pointlist[7] = 10.0;
    in.pointattributelist = (REAL *) malloc(in.numberofpoints *
            in.numberofpointattributes *
            sizeof(REAL));
    in.pointattributelist[0] = 0.0;
    in.pointattributelist[1] = 1.0;
    in.pointattributelist[2] = 11.0;
    in.pointattributelist[3] = 10.0;
    in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
    in.pointmarkerlist[0] = 0;
    in.pointmarkerlist[1] = 2;
    in.pointmarkerlist[2] = 0;
    in.pointmarkerlist[3] = 0;

    in.numberofsegments = 0;
    in.numberofholes = 0;
    in.numberofregions = 1;
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));

    */
    in.regionlist[0] = 0.5;
    in.regionlist[1] = 5.0;
    in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
    in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */

    //printf("Input point set:\n\n");
    //report(&in, 1, 0, 0, 0, 0, 0);

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

    mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of point attributes is zero: */
    mid.pointattributelist = (REAL *) NULL;
    mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
    mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    mid.triangleattributelist = (REAL *) NULL;
    mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
    /* Needed only if segments are output (-p or -c) and -P not used: */
    mid.segmentlist = (int *) NULL;
    /* Needed only if segments are output (-p or -c) and -P and -B not used: */
    mid.segmentmarkerlist = (int *) NULL;
    mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
    mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

    vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
    /* Needed only if -v switch used and number of attributes is not zero: */
    vorout.pointattributelist = (REAL *) NULL;
    vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
    vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

    /* Triangulate the points.  Switches are chosen to read and write a  */
    /*   PSLG (p), preserve the convex hull (c), number everything from  */
    /*   zero (z), assign a regional attribute to each element (A), and  */
    /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
    /*   neighbor list (n).                                              */

//    triangulate("pczAevnQ", &in, &mid, &vorout);
    triangulate("zvnQ", &in, &mid, &vorout);

    printf("searching %i triangles\n",mid.numberoftriangles);
    printf("finding neighbors of point 31\n");

    for (int ee =0; ee < mid.numberoftriangles; ++ee)
        {
        int i1, i2, i3;
        i1 = mid.neighborlist[ee*3];
        i2 = mid.neighborlist[ee*3+1];
        i3 = mid.neighborlist[ee*3+2];
//        printf("%4d  %4d   %4d\n ", mid.neighborlist[ee*3 ],mid.neighborlist[2*ee+1],mid.neighborlist[3*ee+2]);
//        if ( i1==0 || i2==0 ||i3==0)
            printf("%4d %4d %4d \n",i1,i2,i3);
        };


//    printf("Initial triangulation:\n\n");
//    report(&mid, 1, 1, 1, 1, 1, 0);
//    printf("Initial Voronoi diagram:\n\n");
//    report(&vorout, 0, 0, 0, 0, 1, 1);

    /* Attach area constraints to the triangles in preparation for */
    /*   refining the triangulation.                               */

    /* Needed only if -r and -a switches used: */
    mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
    mid.trianglearealist[0] = 3.0;
    mid.trianglearealist[1] = 1.0;

    /* Make necessary initializations so that Triangle can return a */
    /*   triangulation in `out'.                                    */

    out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
    /* Not needed if -N switch used or number of attributes is zero: */
    out.pointattributelist = (REAL *) NULL;
    out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
    /* Not needed if -E switch used or number of triangle attributes is zero: */
    out.triangleattributelist = (REAL *) NULL;

    /* Refine the triangulation according to the attached */
    /*   triangle area constraints.                       */

//    triangulate("pranzBP", &mid, &out, (struct triangulateio *) NULL);

    //printf("Refined triangulation:\n\n");
    //report(&out, 0, 1, 0, 0, 0, 0);

    /* Free all allocated arrays, including those allocated by Triangle. */
    /*
    printf("searching %i triangles\n",mid.numberoftriangles);
    printf("finding neighbors of point 31\n");

    for (int ee =0; ee < out.numberoftriangles; ++ee)
        {
        int i1, i2, i3;
        i1 = out.neighborlist[ee*3];
        i2 = out.neighborlist[ee*3+1];
        i3 = out.neighborlist[ee*3+2];
//        printf("%4d  %4d   %4d\n ", mid.neighborlist[ee*3 ],mid.neighborlist[2*ee+1],mid.neighborlist[3*ee+2]);
        //if ( i1==31 || i2==31 ||i3==31)
            printf("%4d %4d %4d \n",i1,i2,i3);
        };
*/

    free(in.pointlist);
    free(in.pointattributelist);
    free(in.pointmarkerlist);
    free(in.regionlist);
    free(mid.pointlist);
    free(mid.pointattributelist);
    free(mid.pointmarkerlist);
    free(mid.trianglelist);
    free(mid.triangleattributelist);
    free(mid.trianglearealist);
    free(mid.neighborlist);
    free(mid.segmentlist);
    free(mid.segmentmarkerlist);
    free(mid.edgelist);
    free(mid.edgemarkerlist);
    free(vorout.pointlist);
    free(vorout.pointattributelist);
    free(vorout.edgelist);
    free(vorout.normlist);
    free(out.pointlist);
    free(out.pointattributelist);
    free(out.trianglelist);
    free(out.triangleattributelist);

    };

void DelaunayTri::testDel(int numpts, int tmax,bool verbose)
    {
    cout << "Timing Shewchuk's Triangle (9-sheeted)..." << endl;
    nV = numpts;
    float boxa = sqrt(numpts)+1.0;
    box Bx(boxa,boxa);
    setBox(Bx);
    vector<float> ps2(2*numpts);
    float maxx = 0.0;
    int randmax = 1000000;
    for (int i=0;i<numpts;++i)
        {
        float x =EPSILON+boxa/(float)randmax* (float)(rand()%randmax);
        float y =EPSILON+boxa/(float)randmax* (float)(rand()%randmax);
        ps2[i*2]=x;
        ps2[i*2+1]=y;
        //cout <<"{"<<x<<","<<y<<"},";
        };
    setPoints(ps2);

    clock_t tstart,tstop;
    float timing = 0.0;
    tstart = clock();
    for (int tt = 0; tt < tmax; ++tt)
        {
        getTriangulation();
        };
    tstop = clock();
    timing = (tstop-tstart)/(dbl)CLOCKS_PER_SEC/(dbl)tmax;
    cout << "average time per complete triangulation = " << timing<< endl;



    };

