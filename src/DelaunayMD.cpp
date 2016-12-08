#define EPSILON 1e-16
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

using namespace std;

#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"

#include "box.h"

#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.cuh"
#include "gpucell.h"

#include "DelaunayMD.cuh"
#include "DelaunayMD.h"
//DelaunayMD.h included DelaunayCGAL, so gives access to the typedef PDT::Point Point structure



void DelaunayMD::randomizePositions(Dscalar boxx, Dscalar boxy)
    {
    int randmax = 100000000;
    ArrayHandle<Dscalar2> h_points(points,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        Dscalar x =EPSILON+boxx/(Dscalar)(randmax+1)* (Dscalar)(rand()%randmax);
        Dscalar y =EPSILON+boxy/(Dscalar)(randmax+1)* (Dscalar)(rand()%randmax);
        h_points.data[ii].x=x;
        h_points.data[ii].y=y;
        };
    };

void DelaunayMD::reIndexArray(GPUArray<Dscalar2> &array)
    {
    GPUArray<Dscalar2> TEMP = array;
    ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

void DelaunayMD::reIndexArray(GPUArray<Dscalar> &array)
    {
    GPUArray<Dscalar> TEMP = array;
    ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

void DelaunayMD::reIndexArray(GPUArray<int> &array)
    {
    GPUArray<int> TEMP = array;
    ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

void DelaunayMD::spatiallySortPoints()
    {
    //calling this should resort all particle data!
    //
    //itt and tti are the changes that happen in the current sort
    //idxToTag and tagToIdx relate the current indexes to the original ones
    HilbertSorter hs(Box);

    vector<pair<int,int> > idxSorter(N);

    //sort points by Hilbert Curve location
    ArrayHandle<Dscalar2> h_p(points,access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        idxSorter[ii].first=hs.getIdx(h_p.data[ii]);
        idxSorter[ii].second = ii;
        };
    sort(idxSorter.begin(),idxSorter.end());

    //update tti and itt
    for (int ii = 0; ii < N; ++ii)
        {
        int newidx = idxSorter[ii].second;
        itt[ii] = newidx;
        tti[newidx] = ii;
        };

    //update points, idxToTag, and tagToIdx
    vector<int> tempi = idxToTag;
    for (int ii = 0; ii < N; ++ii)
        {
        idxToTag[ii] = tempi[itt[ii]];
        tagToIdx[tempi[itt[ii]]] = ii;
        };
    reIndexArray(points);

    };

void DelaunayMD::resetDelLocPoints()
    {
    ArrayHandle<Dscalar2> h_points(points,access_location::host, access_mode::read);
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
    timestep = 0;
    GPUcompute = true;
    //assorted
    neighMax = 0;
    neighMaxChange = false;
    repPerFrame = 0.0;
    skippedFrames = 0;
    GlobalFixes = 0;
    gputiming = 0.0; cputiming = 0.0;
    //set cellsize to about unity
    cellsize = 1.25;

    //set particle number and box
    N = n;
    Dscalar boxsize = sqrt((Dscalar)N);
    Box.setSquare(boxsize,boxsize);
    CPUbox.setSquare(boxsize,boxsize);

    //set circumcenter array size
    circumcenters.resize(2*(N+10));
    NeighIdxs.resize(6*(N+10));

    points.resize(N);
    pts.resize(N);
    repair.resize(N);
    randomizePositions(boxsize,boxsize);

    //initialize spatial sorting, but do not sort by default
    itt.resize(N);
    tti.resize(N);
    idxToTag.resize(N);
    tagToIdx.resize(N);
    for (int ii = 0; ii < N; ++ii)
        {
        itt[ii]=ii;
        tti[ii]=ii;
        idxToTag[ii]=ii;
        tagToIdx[ii]=ii;
        };

    //cell list initialization
    celllist.setNp(N);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);

    //DelaunayLoc initialization
    box Bx(boxsize,boxsize);
    delLoc.setBox(Bx);
    resetDelLocPoints();

    //make a full triangulation
    FullFails = 1;
    neigh_num.resize(N);
    globalTriangulationCGAL();
    };

void DelaunayMD::updateCellList()
    {
    celllist.setNp(N);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);

    if(GPUcompute)
        {
        celllist.computeGPU(points);
        cudaError_t code = cudaGetLastError();
        if(code!=cudaSuccess)
            {
            printf("cell list computation GPUassert: %s \n", cudaGetErrorString(code));
            throw std::exception();
            };
        }
    else
        {
        vector<Dscalar> psnew(2*N);
        ArrayHandle<Dscalar2> h_points(points,access_location::host, access_mode::read);
        for (int ii = 0; ii < N; ++ii)
            {
            psnew[2*ii] =  h_points.data[ii].x;
            psnew[2*ii+1]= h_points.data[ii].y;
            };
        celllist.setParticles(psnew);
        celllist.compute();

        };

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

void DelaunayMD::reportPos(int i)
    {
    ArrayHandle<Dscalar2> hp(points,access_location::host,access_mode::read);
    printf("particle %i\t{%f,%f}\n",i,hp.data[i].x,hp.data[i].y);
    };

void DelaunayMD::movePointsCPU(GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> h_p(points,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_d(displacements,access_location::host,access_mode::read);
    for (int idx = 0; idx < N; ++idx)
        {
        h_p.data[idx].x += h_d.data[idx].x;
        h_p.data[idx].y += h_d.data[idx].y;
        Box.putInBoxReal(h_p.data[idx]);
        };

    };

void DelaunayMD::movePoints(GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> d_p(points,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_d(displacements,access_location::device,access_mode::readwrite);
    gpu_move_particles(d_p.data,d_d.data,N,Box);
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("movePoints GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };

    };

void DelaunayMD::fullTriangulation()
    {
    resetDelLocPoints();
    cout << "Resetting complete triangulation" << endl;
    //get neighbors of each cell in CW order
    neigh_num.resize(N);

    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::overwrite);

    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);

    vector< vector<int> > allneighs(N);
    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < N; ++nn)
        {
        vector<int> neighTemp;
        delLoc.getNeighbors(nn,neighTemp);
        allneighs[nn]=neighTemp;
        neighnum.data[nn] = neighTemp.size();
        totaln += neighTemp.size();
        if (neighTemp.size() > nmax) nmax= neighTemp.size();
        h_repair.data[nn]=0;
        };
    if (nmax%2 ==0)
        neighMax = nmax + 2;
    else
        neighMax = nmax + 1;
    neighMax = nmax+1; cout << "new Nmax = " << nmax << "; total neighbors = " << totaln << endl;
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
            };
        };

    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("FullTriangulation  GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };

    if(totaln != 6*N)
        {
        printf("CPU neighbor creation failed to match topology! NN = %i \n",totaln);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        getCircumcenterIndices();
        writeTriangulation(output);

        throw std::exception();
        };


    getCircumcenterIndices();
    };


void DelaunayMD::globalTriangulationCGAL(bool verbose)
    {
    GlobalFixes +=1;
    FullFails = 1;
    DelaunayCGAL dcgal;
    ArrayHandle<Dscalar2> h_points(points,access_location::host, access_mode::read);
    vector<pair<Point,int> > Psnew(N);
    for (int ii = 0; ii < N; ++ii)
        {
        Psnew[ii]=make_pair(Point(h_points.data[ii].x,h_points.data[ii].y),ii);
        };
    Dscalar b1,b2,b3,b4;
    Box.getBoxDims(b1,b2,b3,b4);
    dcgal.PeriodicTriangulation(Psnew,b1);

    neigh_num.resize(N);
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);

    int oldNmax = neighMax;
    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < N; ++nn)
        {
        neighnum.data[nn] = dcgal.allneighs[nn].size();
        totaln += dcgal.allneighs[nn].size();
        if (dcgal.allneighs[nn].size() > nmax) nmax= dcgal.allneighs[nn].size();
        h_repair.data[nn]=0;
        };
    if (nmax%2 == 0)
        neighMax = nmax+2;
    else
        neighMax = nmax+1;

    n_idx = Index2D(neighMax,N);
    if(neighMax != oldNmax)
        {
        neighs.resize(neighMax*N);
        neighMaxChange = true;
        };
    updateNeighIdxs();

    //store data in gpuarrays
    {
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::overwrite);

    for (int nn = 0; nn < N; ++nn)
        {
        int imax = neighnum.data[nn];
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,nn);
            ns.data[idxpos] = dcgal.allneighs[nn][ii];
            };
        };

    if(verbose)
        cout << "global new Nmax = " << neighMax << "; total neighbors = " << totaln << endl;cout.flush();
    };

    getCircumcenterIndices(true);

    if(totaln != 6*N)
        {
        printf("global CPU neighbor failed! NN = %i\n",totaln);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        throw std::exception();
        };
    };

void DelaunayMD::updateNeighIdxs()
    {
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int2> h_nidx(NeighIdxs,access_location::host,access_mode::overwrite);
    int idx = 0;
    for (int ii = 0; ii < N; ++ii)
        {
        int nmax = neighnum.data[ii];
        for (int nn = 0; nn < nmax; ++nn)
            {
            h_nidx.data[idx].x = ii;
            h_nidx.data[idx].y = nn;
            idx+=1;
            };
        };
    NeighIdxNum = idx;
    };



void DelaunayMD::getCircumcenterIndices(bool secondtime, bool verbose)
    {
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    ArrayHandle<int3> h_ccs(circumcenters,access_location::host,access_mode::overwrite);

    int totaln = 0;
    int cidx = 0;
    bool fail = false;
    for (int nn = 0; nn < N; ++nn)
        {
        int nmax = neighnum.data[nn];
        totaln+=nmax;
        for (int jj = 0; jj < nmax; ++jj)
            {
            if (fail) continue;

            int n1 = ns.data[n_idx(jj,nn)];
            int ne2 = jj + 1;
            if (jj == nmax-1)  ne2=0;
            int n2 = ns.data[n_idx(ne2,nn)];
            if (nn < n1 && nn < n2)
                {
                h_ccs.data[cidx].x = nn;
                h_ccs.data[cidx].y = n1;
                h_ccs.data[cidx].z = n2;
                cidx+=1;
                };
            };

        };
    NumCircumCenters = cidx;
    if((totaln != 6*N || cidx != 2*N) && !secondtime)
        globalTriangulationCGAL();
    if((totaln != 6*N || cidx != 2*N) && secondtime)
        {
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        printf("step: %i  getCCs failed, %i out of %i ccs, %i out of %i neighs \n",timestep,cidx,2*N,totaln,6*N);
        globalTriangulationCGAL();
        throw std::exception();
        };
    };


void DelaunayMD::repairTriangulation(vector<int> &fixlist)
    {
    int fixes = fixlist.size();
    repPerFrame += ((Dscalar) fixes/(Dscalar)N);
    resetDelLocPoints();

    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::readwrite);

    //First, retriangulate the target points, and check if the neighbor list needs to be reset
    vector<vector<int> > allneighs(fixes);
    bool resetCCidx = false;
    for (int ii = 0; ii < fixes; ++ii)
        {
        int pidx = fixlist[ii];
        vector<int> neighTemp;
        delLoc.getNeighborsCGAL(pidx,neighTemp);
        allneighs[ii]=neighTemp;
        if(neighTemp.size() > neighMax)
            {
            resetCCidx = true;
            };
        };

    //if needed, regenerate the "neighs" structure...hopefully don't do this too much
    if(resetCCidx)
        {
        neighMaxChange = true;
        cout << "Resetting the neighbor structure... new Nmax = "<<neighMax << endl;
        globalTriangulationCGAL();
        return;
        };

    //now, edit the right entries of the neighborlist and neighbor size list
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::readwrite);
    for (int nn = 0; nn < fixes; ++nn)
        {
        int pidx = fixlist[nn];
        int imax = allneighs[nn].size();
        neighnum.data[pidx] = imax;
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,pidx);
            ns.data[idxpos] = allneighs[nn][ii];
            };
        };

    //finally, update the NeighIdx list and Circumcenter list
    updateNeighIdxs();
    getCircumcenterIndices();
    };

void DelaunayMD::testTriangulation()
    {
    //first, update the cell list
    updateCellList();

    //access data handles
    ArrayHandle<Dscalar2> d_pt(points,access_location::device,access_mode::readwrite);

    ArrayHandle<unsigned int> d_cell_sizes(celllist.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(celllist.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
    ArrayHandle<int3> d_ccs(circumcenters,access_location::device,access_mode::read);

    int NumCircumCenters = N*2;
    gpu_test_circumcenters(d_repair.data,
                           d_ccs.data,
                           NumCircumCenters,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           N,
                           celllist.getXsize(),
                           celllist.getYsize(),
                           celllist.getBoxsize(),
                           Box,
                           celllist.cell_indexer,
                           celllist.cell_list_indexer,
                           Fails
                           );
    };

void DelaunayMD::testTriangulationCPU()
    {
    Fails=0;
    if (globalOnly)
        {
        globalTriangulationCGAL();
        skippedFrames -= 1;
        }
    else
        {
        resetDelLocPoints();



        ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
        ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::readwrite);
        ArrayHandle<int> ns(neighs,access_location::host,access_mode::readwrite);
        Fails = 0;
        for (int nn = 0; nn < N; ++nn)
            {
            h_repair.data[nn] = 0;
            vector<int> neighbors;
            for (int ii = 0; ii < neighnum.data[nn];++ii)
                    {
                    int idxpos = n_idx(ii,nn);
                    neighbors.push_back(ns.data[idxpos]);
                    };

            bool good = delLoc.testPointTriangulation(nn,neighbors,false);
            if(!good)
                {
                h_repair.data[nn] = 1;
                Fails=1;
                };
            };
        };
    };


void DelaunayMD::testAndRepairTriangulation(bool verb)
    {
    timestep +=1;

    if (verb) printf("testing triangulation\n");
    if(GPUcompute)
        {
        testTriangulation();
        }
    else
        {
        testTriangulationCPU();
        };
    if(Fails == 1)
        {
        NeedsFixing.clear();
        ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
        if(GPUcompute)
            {
            cudaError_t code = cudaGetLastError();
            if(code!=cudaSuccess)
                {
                printf("testAndRepair preliminary GPUassert: %s \n", cudaGetErrorString(code));
                throw std::exception();
                };
            };

        //add the index and all of its' neighbors
        ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::readwrite);
        ArrayHandle<int> ns(neighs,access_location::host,access_mode::readwrite);
        for (int nn = 0; nn < N; ++nn)
            {
            if (h_repair.data[nn] == 1)
                {
                NeedsFixing.push_back(nn);
                h_repair.data[nn] = 0;
                for (int ii = 0; ii < neighnum.data[nn];++ii)
                    {
                    int idxpos = n_idx(ii,nn);
                    NeedsFixing.push_back(ns.data[idxpos]);
                    };
                };
            };
        sort(NeedsFixing.begin(),NeedsFixing.end());
        NeedsFixing.erase(unique(NeedsFixing.begin(),NeedsFixing.end() ),NeedsFixing.end() );

        if (verb) printf("repairing triangulation via %lu\n",NeedsFixing.size());

        if (NeedsFixing.size() > (N/6))
            {
            FullFails = 1;
            globalTriangulationCGAL();
            }
        else
            {
            FullFails = 0;
            repairTriangulation(NeedsFixing);
            };
        }
    else
        skippedFrames+=1;
    };



void DelaunayMD::readTriangulation(ifstream &infile)
    {
    string line;
    getline(infile,line);
    stringstream convert(line);
    int nn;
    convert >> nn;
    cout << "Reading in " << nn << "points" << endl;
    int idx = 0;
    int ii = 0;
    ArrayHandle<Dscalar2> p(points,access_location::host,access_mode::overwrite);
    while(getline(infile,line))
        {
        Dscalar val = stof(line);
        if (idx == 0)
            {
            p.data[ii].x=val;
            idx +=1;
            }
        else
            {
            p.data[ii].y=val;
            Box.putInBoxReal(p.data[ii]);
            idx = 0;
            ii += 1;
            };
        };
    };

void DelaunayMD::writeTriangulation(ofstream &outfile)
    {
    ArrayHandle<Dscalar2> p(points,access_location::host,access_mode::read);
    outfile << N <<endl;
    for (int ii = 0; ii < N ; ++ii)
        outfile << p.data[ii].x <<"\t" <<p.data[ii].y <<endl;
    };

void DelaunayMD::repel(GPUArray<Dscalar2> &disp,Dscalar eps)
    {
    ArrayHandle<Dscalar2> p(points,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> dd(disp,access_location::host,access_mode::overwrite);
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    Dscalar2 ftot;ftot.x=0.0;ftot.y=0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        Dscalar2 dtot;dtot.x=0.0;dtot.y=0.0;
        Dscalar2 posi = p.data[ii];
        int imax = neighnum.data[ii];
        for (int nn = 0; nn < imax; ++nn)
            {
            int idxpos = n_idx(nn,ii);
            Dscalar2 posj = p.data[ns.data[idxpos]];
            Dscalar2 d;
            Box.minDist(posi,posj,d);

            Dscalar norm = sqrt(d.x*d.x+d.y*d.y);
            if (norm < 1)
                {
                dtot.x-=2*eps*d.x*(1.0-1.0/norm);
                dtot.y-=2*eps*d.y*(1.0-1.0/norm);
                };
            };
        int randmax = 1000000;
        Dscalar xrand = eps*0.1*(-0.5+1.0/(Dscalar)randmax* (Dscalar)(rand()%randmax));
        Dscalar yrand = eps*0.1*(-0.5+1.0/(Dscalar)randmax* (Dscalar)(rand()%randmax));
        dd.data[ii]=dtot;
        ftot.x+=dtot.x+xrand;
        ftot.y+=dtot.y+yrand;
        };
    };
