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

#define DIM 2
#define dbl float
#define REAL float // for triangle
#define EPSILON 1e-12

#include "box.h"
#include "Delaunay1.h"
#include "DelaunayLoc.h"
#include "DelaunayTri.h"

//comment this definition out to compile on cuda-free systems
#define ENABLE_CUDA

#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

#include "DelaunayCheckGPU.h"
#include "DelaunayMD.h"



using namespace std;
using namespace voroguppy;


bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        cout << "Requested GPU (device " << USE_GPU<<") does not exist. Stopping triangulation" << endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)    cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        if (verbose)
            {
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        };
    if (!verbose)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    };


void rnddisp(GPUArray<float2> &disps, int N,float scale)
    {
    disps.resize(N);
    ArrayHandle<float2> h_d(disps,access_location::host,access_mode::overwrite);
    int randmax = 1000000;
    for (int i=0;i<N;++i)
        {
        float x =scale*(-0.5+1.0/(dbl)randmax* (dbl)(rand()%randmax));
        float y =scale*(-0.5+1.0/(dbl)randmax* (dbl)(rand()%randmax));
        h_d.data[i].x=x;
        h_d.data[i].y=y;
        };
    };

int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int c;
    int testRepeat = 5;
    double err = 0.1;

    while((c=getopt(argc,argv,"n:g:m:s:r:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': testRepeat = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'e': err = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };
    clock_t t1,t2;
    
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) return 0;
    cudaSetDevice(USE_GPU);

    DelaunayMD delmd;
    delmd.initialize(numpts);
    delmd.updateCellList();
        delmd.testAndRepairTriangulation();
   
    GPUArray<float2> ds;
    ds.resize(numpts);
    t1=clock();
    for (int tt = 0; tt < testRepeat; ++tt)
        {
        cout << "Starting loop " <<tt << endl;
        rnddisp(ds,numpts,0.001);
        delmd.movePoints(ds);
        delmd.testAndRepairTriangulation();


        };
    t2=clock();
    float movetime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
    cout << "move time (data transfer) ~ " << movetime << " per frame" << endl;



//    delmd.reportCellList();
/*
    float boxa = sqrt(numpts)+1.0;

    box Bx(boxa,boxa);
    gpubox BxGPU(boxa,boxa);
    dbl bx,bxx,by,byy;
    Bx.getBoxDims(bx,bxx,byy,by);
    cout << "Box:" << bx << " " <<bxx << " " <<byy<< " "<< by << endl;


    vector<float> ps2(2*numpts);
    dbl maxx = 0.0;
    int randmax = 1000000;
    for (int i=0;i<numpts;++i)
        {
        float x =EPSILON+boxa/(dbl)randmax* (dbl)(rand()%randmax);
        float y =EPSILON+boxa/(dbl)randmax* (dbl)(rand()%randmax);
        ps2[i*2]=x;
        ps2[i*2+1]=y;
//        cout <<"{"<<x<<","<<y<<"},";
        };
//    cout << endl << maxx << endl;
    cout << endl << endl;

    //simple testing

//    DelaunayNP delnp(ps2);
 //   delnp.testDel(numpts,testRepeat,false);

    DelaunayLoc del(ps2,Bx);
    DelaunayLoc del2(ps2,Bx);
    del.initialize(1.5);
//    del.testDel(numpts,testRepeat,false);
    del2.testDel(numpts,testRepeat,err, true);


//    cout << "Testing cellistgpu" << endl;
//    cellListGPU clgpu2(1.5,ps2,BxGPU);
//    clgpu2.computeGPU();
//
cout << " setting up vector of ccs" << endl;
    vector<int> del_ccs(6*numpts);
    int cid = 0;
    for (int ii = 0; ii < numpts; ++ii)
        {
        vector<int> neighs;
        DelaunayCell cell;
        del.triangulatePoint(ii,neighs,cell,false);
        for (int jj = 0; jj < neighs.size(); ++jj)
            {
            int n1 = neighs[jj];
            int ne2 = jj + 1;
            if (jj == neighs.size()-1) ne2 = 0;
            int n2 = neighs[ne2];
            if (ii < n1 && ii < n2)
                {
                del_ccs[3*cid+0] = ii;
                del_ccs[3*cid+1] = n1;
                del_ccs[3*cid+2] = n2;
                cid+=1;
                };
            };
        };
clock_t t1,t2;
float arraytime = 0.0;
cout << " setting up GPUarrays" << endl;
GPUArray<bool> reTriangulate(numpts);
GPUArray<int> ccs(6*numpts);
t1=clock();
for (int tt = 0; tt < testRepeat; ++tt)
{
  //  cout << "making array of bools" << endl;

    //get gpuarray of bools
    if(true)
        {
        ArrayHandle<bool> tt(reTriangulate,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < numpts; ++ii)
            {
            tt.data[ii]=false;
            };
        };

    if(true)
        ArrayHandle<bool> tt(reTriangulate,access_location::device,access_mode::readwrite);

//    cout << "making array of circumcenter indices" << endl;
    //get gpuarray of circumcenter indices
    if(true)
        {
        ArrayHandle<int> h_ccs(ccs,access_location::host,access_mode::overwrite);
        for (int id = 0; id < 6*numpts; ++id)
            h_ccs.data[id] = del_ccs[id];
        };
    if(true)
        ArrayHandle<int> h_ccs(ccs,access_location::device,access_mode::read);
};
t2=clock();arraytime += (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
cout <<endl << endl << "array conversion time testing time = " << arraytime << endl;



//    cout << "making array of bools" << endl;
    //get gpuarray of bools
*/
/*
    if(true)
        {
        ArrayHandle<bool> tt(reTriangulate,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < numpts; ++ii)
            {
            tt.data[ii]=false;
            };
        };
*/
//    cout << "making array of circumcenter indices" << endl;
    //get gpuarray of circumcenter indices
    //
    //
/*
    if(true)
        {
        ArrayHandle<int> h_ccs(ccs,access_location::host,access_mode::overwrite);
        for (int id = 0; id < 6*numpts; ++id)
            h_ccs.data[id] = del_ccs[id];
        };

cout << "starting GPU test routine" << endl;
t1=clock();


for (int nn = 0; nn < ps2.size(); ++nn)
    {
    float diff = -err*0.5+err*(dbl)(rand()%randmax)/((dbl)randmax); 
    ps2[nn] += diff;
    };
    vector<float> ps3(2*numpts);

float gputime = 0.0;
for (int tt = 0; tt < testRepeat; ++tt)
{

    for (int nn = 0; nn < ps2.size(); ++nn)
        {
        float diff = -err*0.5+err*(dbl)(rand()%randmax)/((dbl)randmax); 
        ps3[nn] = ps2[nn]+ diff;
        };



    t1=clock();
    DelaunayTest gputester;
    gputester.testTriangulation(ps3,ccs,1.25,BxGPU,reTriangulate);
    t2=clock();
    gputime+= (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
//    if (false)
//        {
//        ArrayHandle<bool> h_re(reTriangulate,access_location::host,access_mode::readwrite);
//        for (int nn = 0; nn < numpts; ++nn)
//            {
//            if (h_re.data[nn]) cout << "ah: " <<nn << endl;
//            };
//
//        };
};
//t2=clock();
//float gputime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
cout << "gpu testing time = " << gputime << endl;
cout << "total gpu time = " << gputime + arraytime << endl << endl;
    

cout << "triitesttiming / tritiming = "<< del2.tritesttiming/testRepeat/del2.totaltiming << endl;
cout << "gputest timing  / tritiming = "<< (gputime)/del2.totaltiming << endl;
cout << "gputtotal timing / tritiming = "<< (gputime + arraytime)/del2.totaltiming << endl;
            vector<int> neighs;
            DelaunayCell cell;
            del.triangulatePoint(5,neighs,cell,false);
//            for (int nn = 0; nn < neighs.size(); ++nn)
//                cout << neighs[nn] << "   ";
            cout << endl;
*/
/*
    char fname[256];
    sprintf(fname,"DT.txt");
    ofstream output(fname);
    output.precision(8);
    del.writeTriangulation(output);
    output.close();
*/

    return 0;
};
