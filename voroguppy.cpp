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



/*
#ifdef ENABLE_CUDA
#endif
*/

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

    //
    GPUArray<bool> reTriangulate(numpts);
    if(true)
        {
        ArrayHandle<bool> tt(reTriangulate,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < numpts; ++ii)
            {
            tt.data[ii]=false;
            };

        };



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
