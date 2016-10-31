using namespace std;
#define dbl float
#define EPSILON 1e-12
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

#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"
#include "gpucell.cuh"
#include "gpucell.h"
using namespace voroguppy;


//namespace voroguppy
//{
//DEFINE functions here
cellListGPU::cellListGPU(dbl a, vector<float> &points,gpubox &bx)
    {
    setParticles(points);
    setBox(bx);
    setGridSize(a);
    }
cellListGPU::cellListGPU(vector<float> &points)
    {
    setParticles(points);
    }

void cellListGPU::setParticles(const vector<float> &points)
    {
    int newsize = points.size()/2;
    particles.resize(newsize);
    Np=newsize;
    if(true)
        {
        ArrayHandle<float2> h_handle(particles,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < points.size()/2; ++ii)
            {
            h_handle.data[ii].x = points[2*ii];
            h_handle.data[ii].y = points[2*ii+1];
            };
        };
    };

void cellListGPU::setBox(gpubox &bx)
    {
    dbl b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    Box.setGeneral(b11,b12,b21,b22);
    };

void cellListGPU::setGridSize(dbl a)
    {
    dbl b11,b12,b21,b22;
    Box.getBoxDims(b11,b12,b21,b22);
    xsize = (int)floor(b11/a);
    ysize = (int)floor(b22/a);

    boxsize = b11/xsize;

    xsize = (int)floor(b11/boxsize);
    ysize = (int)floor(b22/boxsize);

    totalCells = xsize*ysize;
    cell_sizes.resize(totalCells); //number of elements in each cell...initialize to zero

    cell_indexer = Index2D(xsize,ysize);
    //estimate Nmax
    Nmax = ceil(Np/totalCells)+1;
    resetCellSizes();
    };

void cellListGPU::resetCellSizes()
    {
    //totalCells=xsize*ysize;
    //cell_sizes.resize(totalCells);
    ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::overwrite);
    for (int cc = 0; cc < totalCells; ++cc)
        h_cell_sizes.data[cc]=0;

    cell_list_indexer = Index2D(Nmax,totalCells);
    idxs.resize(cell_list_indexer.getNumElements());
    ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::overwrite);
    for (int  cc = 0; cc < cell_list_indexer.getNumElements();++cc)
        h_idx.data[cc] = 0;

    assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
    };

void cellListGPU::compute()
    {
    //will loop through particles and put them in cells...
    //if there are more than Nmax particles in any cell, will need to recompute.
    bool recompute = true;
    ArrayHandle<float2> h_pt(particles,access_location::host,access_mode::read);
    int ibin, jbin;
    int nmax = Nmax;
    int computations = 0;
    while (recompute)
        {
        //reset particles per cell, reset cell_list_indexer, resize idxs
        resetCellSizes();
        ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::readwrite);
        ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::readwrite);
        recompute=false;
//cout << "cell list computation #"<<computations << endl;

        for (int nn = 0; nn < Np; ++nn)
            {
            if (recompute) continue;
            ibin = floor(h_pt.data[nn].x/boxsize);
            jbin = floor(h_pt.data[nn].y/boxsize);

            int bin = cell_indexer(ibin,jbin);
//cout << bin << "out of bins " << totalCells << "for particle " << nn << endl; cout.flush();
            int offset = h_cell_sizes.data[bin];
//cout << "offset = " << offset <<  "  cli is " << cell_list_indexer(offset,bin)<< endl; cout.flush();
            if (offset < Nmax)
                {
                int clpos = cell_list_indexer(offset,bin);
                h_idx.data[cell_list_indexer(offset,bin)]=nn;
//            cout << "particle " << nn << " ibin =" << ibin<< " jbin=" <<jbin << " bin "<<bin << "out of " << totalCells<< "  Nmax = " << Nmax << "  clpos = " << clpos << " OFFSET " << offset<< endl;
 //               cout.flush();
                }
            else
                {
                nmax = max(Nmax,offset+1);
                Nmax=nmax;
                recompute=true;
                };
            h_cell_sizes.data[bin]++;
            };
        computations++;
        };
    cell_list_indexer = Index2D(Nmax,totalCells);
    };


void cellListGPU::computeGPU()
    {
    bool recompute = true;
    resetCellSizes();

    while (recompute)
        {
        //cout << "computing cell list on the gpu with Nmax = " << Nmax << endl;
        resetCellSizes();

        //scope for arrayhandles
        if (true)
            {
            //get particle data
            ArrayHandle<float2> d_pt(particles,access_location::device,access_mode::read);

            //get cell list arrays...readwrite so things are properly zeroed out
            ArrayHandle<unsigned int> d_cell_sizes(cell_sizes,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_idx(idxs,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);

            //call the gpu function
            gpu_compute_cell_list(d_pt.data,        //particle positions...broken
                          d_cell_sizes.data,//particles per cell
                          d_idx.data,       //cell list
                          Np,               //number of particles
                          Nmax,             //maximum particles per cell
                          xsize,            //number of cells in x direction
                          ysize,            // ""     ""      "" y directions
                          boxsize,          //size of each grid cell
                          Box,
                          cell_indexer,
                          cell_list_indexer,
                          d_assist.data
                          );               //the box
            }
        //get cell list arrays
        recompute = false;
        //bool loopcheck=false;
        if (true)
            {
/*
            ArrayHandle<int> h_as(assist,access_location::host,access_mode::read);
            cout << Nmax << endl;
            cout << h_as.data[0]<< "   " << h_as.data[1] << endl;
            if (h_as.data[1]==1)
                {
                Nmax=h_as.data[0];
                recompute=true;
                };

*/

            ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::read);
            ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::read);
            for (int cc = 0; cc < totalCells; ++cc)
                {
                //if (loopcheck) continue;
                int cs = h_cell_sizes.data[cc] ;
                for (int bb = 0; bb < cs; ++bb)
                    {
                    int wp = cell_list_indexer(bb,cc);
  //                  cout <<" cell " <<cc << "pp  "<<h_idx.data[wp] <<endl;
                    };
                if(cs > Nmax)
                    {
                    Nmax =cs ;
                    recompute = true;
                    //cout << cs <<"in cell " << cc << endl;
                    //loopcheck = true;
                    };

                };

            };
        };
    cell_list_indexer = Index2D(Nmax,totalCells);
//    cout << "Nmax = " << Nmax << endl;

    };




//} //end namespace

