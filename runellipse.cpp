#include "std_include.h"

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
#define EPSILON 1e-16

#include "box.h"
#include "Delaunay1.h"
#include "DelaunayLoc.h"

//#include "DelaunayCGAL.h"

//comment this definition out to compile on cuda-free systems
#define ENABLE_CUDA

#include "Matrix.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

#include "DelaunayMD.h"
#include "spv2d.h"


#include "Database.h"

using namespace std;


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
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl;
        };
    return true;
    };



int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int USE_TENSION = 0;
    int c;
    int tSteps = 5;
    int initSteps = 0;

    Dscalar dt = 0.1;
    Dscalar p0 = 4.0;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;
    Dscalar Dr = 1.0;
    Dscalar gamma = 0.0;

    int program_switch = 0;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:d:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'x': USE_TENSION = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
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



    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) return 0;
    cudaSetDevice(USE_GPU);



    char dataname[256];
    sprintf(dataname,"../../data/spv/Ellipse/Ellipse_N%i_p%.3f_v%.3f_Dr%.3f_g%.3f.nc",numpts,p0,v0,Dr,gamma);
    SPVDatabase ncdat(numpts,dataname,NcFile::Replace);

    SPV2D spv(numpts,1.0,p0);

    spv.setv0Dr(v0,Dr);
    spv.setDeltaT(dt);


    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };


    printf("Setting cells within the central ellipse to different type, applying tension...\n");
    spv.setCellTypeEllipse(0.25,2.0);
    spv.setUseTension(true);
    spv.setTension(gamma);
    spv.reportCellInfo();
    spv.setSortPeriod(10000);
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10000 ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(spv);
            };
        spv.performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl <<endl <<endl;

    return 0;
};
