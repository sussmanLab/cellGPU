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

//#define DIM 2
#define dbl float
#define REAL float // for triangle
#define EPSILON 1e-12

#include "box.h"
#include "Delaunay1.h"
#include "DelaunayLoc.h"
#include "DelaunayTri.h"

#include "DelaunayCGAL.h"

//comment this definition out to compile on cuda-free systems
#define ENABLE_CUDA

#include "Matrix.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

#include "DelaunayCheckGPU.h"
#include "DelaunayMD.h"
#include "spv2d.h"



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
    float p0 = 4.0;
    float a0 = 1.0;
    float v0 = 0.1;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': testRepeat = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'e': err = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
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

    char fname[256];
    char fname0[256];
    char fname1[256];
    char fname2[256];
    sprintf(fname0,"DT0.txt");
    sprintf(fname1,"DT1.txt");
    sprintf(fname2,"DT2.txt");
    ofstream output0(fname0);
    output0.precision(8);
    ofstream output1(fname1);
    output1.precision(8);
    ofstream output2(fname2);
    output2.precision(8);

    vector<int> cts(numpts);
    for (int ii = 0; ii < numpts; ++ii) 
        {
        if(ii < numpts/2)
            cts[ii]=0;
        else
            cts[ii]=1;
        };

    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) return 0;
    cudaSetDevice(USE_GPU);



    SPV2D spv(numpts,1.0,p0);
    spv.writeTriangulation(output0);
/*
    //Compare force with output of Mattias' code
    char fn[256];
    sprintf(fn,"/hdd2/repos/test.txt");
    ifstream input(fn);
    spv.readTriangulation(input);
    spv.globalTriangulation();
    spv.setCellPreferencesUniform(a0,p0);
    spv.computeGeometry();
      spv.setCellType(cts);
    for (int ii = 0; ii < numpts; ++ii)
        {
        //spv.computeSPVForceCPU(ii);
        spv.computeSPVForceWithTensionsCPU(ii,.2);
        };
    spv.reportForces();
    cout << "current q = " << spv.reportq() << endl;
    spv.meanForce();



    for(int ii = 0; ii < 100; ++ii)
        {
    //    spv.performTimestep();
        };
//    cout << "current q = " << spv.reportq() << endl;
*/
    spv.writeTriangulation(output1);
    spv.setCellPreferencesUniform(1.0,p0);
    spv.setDeltaT(err);
    spv.setv0(v0);


    //cts is currently 0 for first half, 1 for second half
    spv.setCellType(cts);
/*
    if(true)
        {
        ArrayHandle<float2> h_p(spv.points,access_location::host,access_mode::read);
        for (int ii = 0; ii < numpts; ++ii)
            printf("(%f\t%f)\n",h_p.data[ii].x,h_p.data[ii].y);
        };
*/

//    spv.performTimestep();

    t1=clock();
    for(int ii = 0; ii < testRepeat; ++ii)
        {
//        vector<int> nes;
//        spv.delLoc.getNeighborsTri(602,nes);
//        for (int jj = 0; jj < nes.size(); ++jj) printf("%i\t",nes[jj]);
//        printf("\n");
        spv.performTimestep();

        if(ii%100 ==0)
//if(true)
            {
            printf("timestep %i\n",ii);
//            spv.meanForce();
            char fn[256];
            sprintf(fn,"/hdd2/data/spv/bidisperse/DTg2%i.txt",ii);
            ofstream outputc(fn);
            output1.precision(8);
            spv.writeTriangulation(outputc);
            };
        };
    t2=clock();
    float steptime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
    cout << "timestep ~ " << steptime << " per frame; " << spv.repPerFrame/testRepeat*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl;
    cout << "current q = " << spv.reportq() << endl;



/*
    t1=clock();
    for(int ii = 0; ii < testRepeat; ++ii)
        {
        if(ii%100 ==0) printf("timestep %i\n",ii);
        spv.performTimestep();
        };
    t2=clock();
    steptime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
    cout << "timestep ~ " << steptime << " per frame; " << spv.repPerFrame/2./testRepeat*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl;
    cout << "current q = " << spv.reportq() << endl;
*/
    spv.writeTriangulation(output2);

    //spv.computeGeometryCPU();
    //for (int ii = 0; ii < numpts; ++ii) spv.computeSPVForceCPU(ii);
    //spv.meanForce();
    //for (int tt = 0; tt < testRepeat; ++tt) spv.performTimestep();

    //spv.computeGeometryCPU();

    //for (int ii = 0; ii < numpts; ++ii) spv.computeSPVForceCPU(ii);
    //spv.meanForce();
    //spv.meanArea();
    //spv.computeGeometryCPU();
    //for (int ii = 0; ii < numpts; ++ii) spv.computeSPVForceCPU(ii);
    //spv.meanForce();

    /*
    t1=clock();
    for (int ii = 0; ii < testRepeat;++ii)
        spv.computeGeometry();
    t2=clock();
    cout << "geometry timing ~ " << (t2-t1)/(dbl)CLOCKS_PER_SEC << endl;
    spv.meanArea();

    t1=clock();
    for (int ii = 0; ii < testRepeat;++ii)
        spv.computeGeometryCPU();
    t2=clock();
    cout << "geometryCPU timing ~ " << (t2-t1)/(dbl)CLOCKS_PER_SEC << endl;
    */


/*
    float boxa = sqrt(numpts);

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
//        cout <<i << "  {"<<x<<","<<y<<"},";
        };
//    cout << endl << maxx << endl;
    cout << endl << endl;

    //simple testing
    //
    DelaunayCGAL dcgal;
    t1=clock();
    for (int jj = 0; jj < testRepeat; ++jj)
        dcgal.PeriodicTriangulation(ps2,boxa);
    t2=clock();
    float cgaltime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
    cout <<endl << endl << "CGAL time  = " << cgaltime << endl;
    //
    DelaunayTri dtri(ps2);
    dtri.getTriangulation();
    vector<int> nes;
    dtri.getNeighbors(ps2,31,nes);
    dtri.getNeighbors(ps2,21,nes);

    DelaunayLoc del(ps2,Bx);
    del.initialize(1.5);
    vector<int> neighs;
    DelaunayCell cell;
    del.triangulatePoint(31,neighs,cell,false);
    cout << " DelLoc neighbors:" << endl;
    for (int ii = 0; ii < neighs.size(); ++ii)
        {
        printf("%i \t",neighs[ii]);
        };
    printf("\n");
    del.triangulatePoint(46,neighs,cell,false);
    cout << " DelLoc neighbors:" << endl;
    for (int ii = 0; ii < neighs.size(); ++ii)
        {
        printf("%i \t",neighs[ii]);
        printf("(%f,%f) \n",ps2[2*neighs[ii]],ps2[2*neighs[ii]+1]);
        };
    printf("\n");

    del.getNeighborsTri(21,neighs);
    cout << " DelLoc neighbors:" << endl;
    for (int ii = 0; ii < neighs.size(); ++ii)
        printf("%i \t",neighs[ii]);
    printf("\n");


*/

    /*

    DelaunayMD delmd;
    delmd.initialize(numpts);
//    delmd.updateCellList();


//    delmd.testAndRepairTriangulation();
    delmd.writeTriangulation(output1);
    if(numpts < 600) delmd.setCPU();
    GPUArray<float2> ds,ps;
   
    ds.resize(numpts);
    t1=clock();
    for (int tt = 0; tt < testRepeat; ++tt)
        {
        if (tt % 1000 ==0) 
            {
            cout << "Starting loop " <<tt << endl;
            //delmd.fullTriangulation();
            };
        if(tt%2 == 0)
            delmd.randmax = 1000000;
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
