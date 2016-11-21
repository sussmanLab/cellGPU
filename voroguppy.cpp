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
#include "cuda_profiler_api.h"
#include "vector_types.h"

//#define DIM 2
#define dbl float
#define REAL float // for triangle
#define EPSILON 1e-12

#include "box.h"
#include "Delaunay1.h"
#include "DelaunayLoc.h"
#include "DelaunayTri.h"

//#include "DelaunayCGAL.h"

//comment this definition out to compile on cuda-free systems
#define ENABLE_CUDA

#include "Matrix.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

#include "DelaunayCheckGPU.h"
#include "DelaunayMD.h"
#include "spv2d.h"


#include "Database.h"

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



int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int USE_TENSION = 0;
    int c;
    int testRepeat = 5;
    int initSteps = 0;
    double err = 0.1;
    float p0 = 4.0;
    float a0 = 1.0;
    float v0 = 0.1;
    float gamma = 0.0;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': testRepeat = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'x': USE_TENSION = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': err = atof(optarg); break;
            case 's': gamma = atof(optarg); break;
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

    char dataname[256];
    sprintf(dataname,"/hdd2/data/spv/Ellipse_N%i_p%.2f_g%.2f.nc",numpts,p0,gamma);
    SPVDatabase ncdat(numpts,dataname,NcFile::Replace);

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
    spv.setTension(gamma);
    if(USE_TENSION != 0) spv.setUseTension(true);


    spv.setCellPreferencesUniform(1.0,p0);
    spv.setDeltaT(err);
    spv.setv0(v0);

    //cts is currently 0 for first half, 1 for second half


    spv.setCellType(cts);
    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };

    printf("Finished with initialization\n");

    printf("Setting cells within the central ellipse to different type, applying tension...\n");
    spv.setCellTypeEllipse(0.25,2.0);
    spv.setUseTension(true);
    spv.setTension(gamma);

cudaProfilerStart();
    t1=clock();
    for(int ii = 0; ii < testRepeat; ++ii)
        {

        if(ii%10000 ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(spv);
            };
        spv.performTimestep();
        };
    t2=clock();
cudaProfilerStop();
    float steptime = (t2-t1)/(dbl)CLOCKS_PER_SEC/testRepeat;
    cout << "timestep ~ " << steptime << " per frame; " << spv.repPerFrame/testRepeat*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl;
    cout << "current q = " << spv.reportq() << endl;
    spv.meanForce();

    cout << endl << "force time  = " << spv.forcetiming/(float)CLOCKS_PER_SEC/(initSteps+testRepeat) << endl;
    cout << "other time  = " << spv.triangletiming/(float)CLOCKS_PER_SEC/(initSteps+testRepeat) << endl;



    return 0;
};
