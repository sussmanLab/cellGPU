using namespace std;
#define EPSILON 1e-12
#define dbl float
#define ENABLE_CUDA

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

//#include "cuda.h"
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"

#include "box.h"

#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.cuh"
#include "gpucell.h"

#include "DelaunayMD.h"



void DelaunayMD::randomizePositions(float boxx, float boxy)
    {
    int randmax = 100000000;
    ArrayHandle<float2> h_points(points,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        float x =EPSILON+boxx/(dbl)(randmax+1)* (dbl)(rand()%randmax);
        float y =EPSILON+boxy/(dbl)(randmax+1)* (dbl)(rand()%randmax);
        h_points.data[ii].x=x;
        h_points.data[ii].y=y;
        };
    };

void DelaunayMD::resetDelLocPoints()
    {
    ArrayHandle<float2> h_points(points,access_location::host, access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        pts[ii].x=h_points.data[ii].x;
        pts[ii].y=h_points.data[ii].y;
        };
    delLoc.setPoints(pts);
    delLoc.initialize(cellsize);

    };

void DelaunayMD::initialize(int n)
    {
    //set cellsize to about unity
    cellsize = 1.25;

    //set particle number and box
    N = n;
    float boxsize = sqrt(N);
    Box.setSquare(boxsize,boxsize);

    //set particle positions (randomly)
    points.resize(N);
    pts.resize(N);
    randomizePositions(boxsize,boxsize);

    //cell list initialization
    celllist.setNp(N);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);

    //DelaunayLoc initialization
    box Bx(boxsize,boxsize);
    delLoc.setBox(Bx);
    resetDelLocPoints();

    //make a full triangulation
    fullTriangulation();
    };

void DelaunayMD::updateCellList()
    {
    celllist.computeGPU(points);
    };

void DelaunayMD::reportCellList()
    {
    ArrayHandle<unsigned int> h_cell_sizes(celllist.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> h_idx(celllist.idxs,access_location::host,access_mode::read);
    int numCells = celllist.getXsize()*celllist.getYsize();
    for (int nn = 0; nn < numCells; ++nn)
        {
        cout << "cell " <<nn <<":     ";
        for (int offset = 0; offset < h_cell_sizes.data[nn]; ++offset)
            {
            int clpos = celllist.cell_list_indexer(offset,nn);
            cout << h_idx.data[clpos] << "   ";
            };
        cout << endl;

        };

    };

void DelaunayMD::fullTriangulation()
    {
    cout << "Resetting complete triangulation" << endl;
    //get neighbors of each cell in CW order
    neigh_num.resize(N);
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::overwrite);
    vector< vector<int> > allneighs(N);
    int nmax = 0;
    for(int nn = 0; nn < N; ++nn)
        {
        vector<int> neighTemp;
        delLoc.getNeighbors(nn,neighTemp);
        allneighs[nn]=neighTemp;
        neighnum.data[nn] = neighTemp.size();
        if (neighTemp.size() > nmax) nmax= neighTemp.size();
        };
    neighMax = nmax; cout << "new Nmax = " << nmax << endl;
    neighs.resize(neighMax*N);

    //store data in gpuarray
    n_idx = Index2D(neighMax,N);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::overwrite);
    for (int nn = 0; nn < N; ++nn)
        {
        int imax = neighnum.data[nn];
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,nn);
            ns.data[idxpos] = allneighs[nn][ii];
//printf("particle %i (%i,%i)\n",nn,idxpos,allneighs[nn][ii]);
            };

        };

    };

