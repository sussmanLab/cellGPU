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

#include "cuda_runtime.h"

#define DIM 2
#define dbl float
#define REAL float // for triangle
#define EPSILON 1e-12

#include "box.h"
#include "Delaunay1.h"
#include "DelaunayLoc.h"
#include "DelaunayTri.h"

#include "DelaunayCheckGPU.h"



using namespace std;
using namespace voroguppy;

int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int c;
    int testRepeat = 5;

    while((c=getopt(argc,argv,"n:g:m:s:r:b:x:y:z:p:t:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': testRepeat = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
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
        //cout <<"{"<<x<<","<<y<<"},";
        };
//    cout << endl << maxx << endl;
    cout << endl << endl;

    //simple testing

//    DelaunayNP delnp(ps2);
 //   delnp.testDel(numpts,testRepeat,false);

    DelaunayLoc del(ps2,Bx);
//    del.testDel(numpts,testRepeat,false);
    del.testDel(numpts,1,false);


    cout << "Testing cellistgpu" << endl;
    cellListGPU clgpu2(1.5,ps2,BxGPU);
    clgpu2.computeGPU();

    cout << "making array of bools" << endl;

    //get gpuarray of bools
    GPUArray<bool> reTriangulate(numpts);
    if(true)
        {
        ArrayHandle<bool> tt(reTriangulate,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < numpts; ++ii)
            {
            tt.data[ii]=false;
            };
        };
    cout << "making array of circumcenter indices" << endl;
    //get gpuarray of circumcenter indices
    GPUArray<int> ccs(6*numpts);
    if(true)
        {
        ArrayHandle<int> h_ccs(ccs,access_location::host,access_mode::overwrite);
        int cidx = 0;
        for (int nn = 0; nn < numpts; ++nn)
            {
            vector<int> neighs;
            DelaunayCell cell;
            del.triangulatePoint(nn,neighs,cell,false);
            for (int jj = 0; jj < neighs.size();++jj)
                {
                int n1 = neighs[jj];
                int ne2 = jj + 1;
                if (jj == neighs.size()-1) ne2 = 0;
                int n2 = neighs[ne2];
                if (nn < n1 && nn < n2)
                    {
                    h_ccs.data[3*cidx+0] = nn;
                    h_ccs.data[3*cidx+1] = n1;
                    h_ccs.data[3*cidx+2] = n2;
                    cidx+=1;
                    };
                };
            };
        };
cout << "starting GPU test routine" << endl;
clock_t t1,t2;
t1=clock();
for (int tt = 0; tt < testRepeat; ++tt)
{
    DelaunayTest gputester;
    gputester.testTriangulation(ps2,ccs,1.25,BxGPU,reTriangulate);
};
t2=clock();
float gputime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
cout << "gpu testing time = " << gputime << endl;


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
