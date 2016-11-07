using namespace std;
#define EPSILON 1e-12
#define dbl float
#define REAL double
#define ANSI_DECLARATIONS
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

#include "DelaunayTri.h"
#include "DelaunayMD.cuh"
#include "DelaunayMD.h"



void DelaunayMD::randomizePositions(float boxx, float boxy)
    {
    int randmax = 100000000;
    ArrayHandle<float2> h_points(points,access_location::host, access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        float x =EPSILON+boxx/(float)(randmax+1)* (float)(rand()%randmax);
        float y =EPSILON+boxy/(float)(randmax+1)* (float)(rand()%randmax);
        h_points.data[ii].x=x;
        h_points.data[ii].y=y;
//        printf("%i; {%f,%f}\n",ii,x,y);
        };
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("randomizePos GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
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
    timestep = 0;
    GPUcompute = true;
    //assorted
    neighMax = 0;
    repPerFrame = 0.0;
    //set cellsize to about unity
    cellsize = 1.25;

    //set particle number and box
    N = n;
    float boxsize = sqrt(N);
    Box.setSquare(boxsize,boxsize);
    CPUbox.setSquare(boxsize,boxsize);

    //set circumcenter array size
    circumcenters.resize(6*(N+10));

    //set particle positions (randomly)
    points.resize(N);
    pts.resize(N);
    repair.resize(N);
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
    globalTriangulation();
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("delMD initialization GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };
    };

void DelaunayMD::updateCellList()
    {
    celllist.setNp(N);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);

    cudaError_t code1 = cudaGetLastError();
    if(code1!=cudaSuccess)
        {
        printf("cell list preliminary computation GPUassert: %s \n", cudaGetErrorString(code1));
        throw std::exception();
        };


    celllist.computeGPU(points);
    //celllist.compute(points);
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("cell list computation GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
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
    ArrayHandle<float2> hp(points,access_location::host,access_mode::read);
    printf("particle %i\t{%f,%f}\n",i,hp.data[i].x,hp.data[i].y);
    };

void DelaunayMD::movePoints(GPUArray<float2> &displacements)
    {
    ArrayHandle<float2> d_p(points,access_location::device,access_mode::readwrite);
    ArrayHandle<float2> d_d(displacements,access_location::device,access_mode::readwrite);
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
        delLoc.getNeighborsTri(nn,neighTemp);
        allneighs[nn]=neighTemp;
        neighnum.data[nn] = neighTemp.size();
        totaln += neighTemp.size();
        if (neighTemp.size() > nmax) nmax= neighTemp.size();
        h_repair.data[nn]=0;
        };
    neighMax = nmax; cout << "new Nmax = " << nmax << "; total neighbors = " << totaln << endl;
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

    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("FullTriangulation  GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };

    if(totaln != 6*N)
        {
        printf("CPU neighbor creation failed to match topology! NN = %i \n",totaln);
//        ArrayHandle<float2> p(points,access_location::host,access_mode::read);
//        for (int ii = 0; ii < N; ++ii)
//            printf("(%f,%f)\n",p.data[ii].x,p.data[ii].y);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        getCircumcenterIndices();
        writeTriangulation(output);
            
        throw std::exception();
        };


    getCircumcenterIndices();
    };


void DelaunayMD::globalTriangulation()
    {
    cout << "Resetting complete triangulation globally" << endl;

    //get neighbors of each cell in CW order from the Triangle interface
    vector<float> psnew(2*N);
    ArrayHandle<float2> h_points(points,access_location::host, access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        psnew[2*ii] = h_points.data[ii].x;
        psnew[2*ii+1]=h_points.data[ii].y;
        };
    vector< vector<int> > allneighs(N);
    DelaunayTri delTri;
    delTri.fullPeriodicTriangulation(psnew,CPUbox,allneighs);

    neigh_num.resize(N);

    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);

    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < N; ++nn)
        {
        neighnum.data[nn] = allneighs[nn].size();
        totaln += allneighs[nn].size();
        if (allneighs[nn].size() > nmax) nmax= allneighs[nn].size();
        h_repair.data[nn]=0;
        };
    neighMax = nmax; cout << "global new Nmax = " << nmax << "; total neighbors = " << totaln << endl;
    neighs.resize(neighMax*N);
    n_idx = Index2D(neighMax,N);

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

    getCircumcenterIndices(true);

    if(totaln != 6*N)
        {
        printf("global CPU neighbor failed! NN = %i\n",totaln);
//        ArrayHandle<float2> p(points,access_location::host,access_mode::read);
//        for (int ii = 0; ii < N; ++ii)
//            printf("(%f,%f)\n",p.data[ii].x,p.data[ii].y);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
//        throw std::exception();
        };

    };




void DelaunayMD::getCircumcenterIndices(bool secondtime)
    {
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    ArrayHandle<int> h_ccs(circumcenters,access_location::host,access_mode::overwrite);

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
//if(nn == 20 || n1 ==20 || n2 == 20) printf("%i %i %i\n",nn,n1,n2);
            if (nn < n1 && nn < n2)
                {
//                if (fail) {cidx +=1;continue;};
//                if (cidx == 2*N) fail = true;
                h_ccs.data[3*cidx] = nn;
                h_ccs.data[3*cidx+1] = n1;
                h_ccs.data[3*cidx+2] = n2;
                cidx+=1;
                };
            };

        };
    NumCircumCenters = cidx;
  //  if (totaln != 6*N || fail || cidx > 3*N) fullTriangulation();
//    cout << "Number of ccs processed : " << cidx << " with total neighbors "<< totaln << endl;
    if((totaln != 6*N || cidx != 2*N) && !secondtime)
        {
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        printf("step: %i  getCCs failed, %i out of %i ccs, %i out of %i neighs \n",timestep,cidx,2*N,totaln,6*N);
        globalTriangulation();
//        throw std::exception();
        };


    //cout << "Number of ccs processed : " << cidx << " with total neighbors "<< totaln << endl;
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("getCCIndices GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };

    };


void DelaunayMD::repairTriangulation(vector<int> &fixlist)
    {
    int fixes = fixlist.size();
    //if there is nothing to fix, skip this routing (and its expensive memory accesses)
    repPerFrame += ((float) fixes/(float)N);
    if (fixes == 0) return;
//    cout << "about to repair " << fixes << " points" << endl;
    resetDelLocPoints();

    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::readwrite);

    //First, retriangulate the target points, and check if the neighbor list needs to be reset 
    //(if neighMax changes)
    //
    //overwrite the first fixes elements of allneighs to save on vector costs, or something?
    vector<vector<int> > allneighs(fixes);
    bool resetCCidx = false;
    for (int ii = 0; ii < fixes; ++ii)
        {
        int pidx = fixlist[ii];
        vector<int> neighTemp;
        delLoc.getNeighborsTri(pidx,neighTemp);
        allneighs[ii]=neighTemp;
        if(neighTemp.size() > neighMax)
            {
            neighMax = neighTemp.size();
            resetCCidx = true;
            };
        };
    
    //if needed, regenerate the "neighs" structure...hopefully don't do this too much
    //Also, think about occasionally shrinking the list if it is much too big?
    if(resetCCidx)
        {
        cout << "Resetting the neighbor structure... new Nmax = "<<neighMax << endl;
        globalTriangulation();
        return;
        };

    //now, edit the right entries of the neighborlist and neighbor size list
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::readwrite);
    for (int nn = 0; nn < fixes; ++nn)
        {
        int pidx = fixlist[nn];
        int imax = allneighs[nn].size();
        neighnum.data[pidx] = imax;
//        cout << " particle " << pidx << " neighs = " << imax << endl;
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,pidx);
            ns.data[idxpos] = allneighs[nn][ii];
//            cout << ns.data[idxpos] << "    ";
            };
//        cout << endl;
        };

    getCircumcenterIndices();
    };

void DelaunayMD::testTriangulation()
    {
    //first, update the cell list
    updateCellList();

    //access data handles
    ArrayHandle<float2> d_pt(points,access_location::device,access_mode::readwrite);

    ArrayHandle<unsigned int> d_cell_sizes(celllist.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(celllist.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_ccs(circumcenters,access_location::device,access_mode::read);

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
                           celllist.cell_list_indexer
                           );
    };

void DelaunayMD::testTriangulationCPU()
    {
    resetDelLocPoints();



    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
    
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::readwrite);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::readwrite);

    for (int nn = 0; nn < N; ++nn)
        {
        vector<int> neighbors;
        for (int ii = 0; ii < neighnum.data[nn];++ii)
                {
                int idxpos = n_idx(ii,nn);
                neighbors.push_back(ns.data[idxpos]);
                };
        
        bool good = delLoc.testPointTriangulation(nn,neighbors,false);
        if(!good) h_repair.data[nn]=1;
        };

    };


void DelaunayMD::testAndRepairTriangulation()
    {
    timestep +=1;
    if(GPUcompute)
        testTriangulation();
    else
        testTriangulationCPU();
    vector<int> NeedsFixing;
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("testAndRepair preliminary GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
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
//                printf("testing %i\t ",ns.data[idxpos]);
                };
            };
        };
       sort(NeedsFixing.begin(),NeedsFixing.end());
       NeedsFixing.erase(unique(NeedsFixing.begin(),NeedsFixing.end() ),NeedsFixing.end() );
       repairTriangulation(NeedsFixing);
    };

void DelaunayMD::writeTriangulation(ofstream &outfile)
    {
    ArrayHandle<float2> p(points,access_location::host,access_mode::read);
    ArrayHandle<int> cc(circumcenters,access_location::host,access_mode::read);
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    outfile << N <<endl;
    for (int ii = 0; ii < N ; ++ii)
        outfile << p.data[ii].x <<"\t" <<p.data[ii].y <<endl;
    for (int ii = 0; ii < 2*N; ++ii)
        outfile << cc.data[3*ii] <<"\t" <<cc.data[3*ii+1]<<"\t"<<cc.data[3*ii+2]<<endl;
    for (int ii = 0; ii < N; ++ii)
        {
        int imax = neighnum.data[ii];
        for (int nn = 0; nn < imax; ++nn)
            {
            int idxpos = n_idx(nn,ii);
            outfile << ns.data[idxpos] << "\t";
            };
        outfile << endl;
        };
    };

void DelaunayMD::repel(GPUArray<float2> &disp,float eps)
    {
    ArrayHandle<float2> p(points,access_location::host,access_mode::read);
    ArrayHandle<float2> dd(disp,access_location::host,access_mode::overwrite);
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    float2 ftot;ftot.x=0.0;ftot.y=0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        float2 dtot;dtot.x=0.0;dtot.y=0.0;
        float2 posi = p.data[ii];
        int imax = neighnum.data[ii];
        for (int nn = 0; nn < imax; ++nn)
            {
            int idxpos = n_idx(nn,ii);
            float2 posj = p.data[ns.data[idxpos]];
            float2 d;
            Box.minDist(posi,posj,d);

            float norm = sqrt(d.x*d.x+d.y*d.y);
            if (norm < 1)
                {
                dtot.x-=2*eps*d.x*(1.0-1.0/norm);
                dtot.y-=2*eps*d.y*(1.0-1.0/norm);
                };
            };
        int randmax = 1000000;
        float xrand = eps*0.1*(-0.5+1.0/(dbl)randmax* (dbl)(rand()%randmax));
        float yrand = eps*0.1*(-0.5+1.0/(dbl)randmax* (dbl)(rand()%randmax));
        dd.data[ii]=dtot;
        ftot.x+=dtot.x+xrand;
        ftot.y+=dtot.y+yrand;
        };
//    printf("Total force = (%f,%f)\n",ftot.x,ftot.y);
    };
