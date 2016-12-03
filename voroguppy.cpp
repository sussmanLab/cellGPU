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


#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "vector_types.h"

#define ENABLE_CUDA
#define EPSILON 1e-16


#include "spv2d.h"
//#include "Database.h"


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
    int tSteps = 5;
    int initSteps = 0;

    Dscalar dt = 0.1;
    Dscalar p0 = 4.0;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;
    Dscalar gamma = 0.0;

    int program_switch = 0;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'x': USE_TENSION = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
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


    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
//    char dataname[256];
//    sprintf(dataname,"/hdd2/data/spv/test.nc");
//    SPVDatabase ncdat(numpts,dataname,NcFile::Replace);


//    cudaProfilerStart();

    SPV2D spv(numpts,1.0,p0);
    if (USE_GPU < 0)
        spv.setCPU();

    spv.setCellPreferencesUniform(1.0,p0);
    spv.setv0Dr(v0,1.0);
    spv.setDeltaT(dt);


    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };
    spv.meanForce();

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    //spv.meanForce();
    spv.repPerFrame = 0.0;

    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10000 ==0)
            {
            printf("timestep %i\n",ii);
            };
        spv.performTimestep();
        };
    //cudaProfilerStop();
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl << spv.skippedFrames << " skipped frames" << endl << endl;

    cout << endl << "force time  = " << spv.forcetiming/(Dscalar)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;
    cout << "triangle time  = " << spv.triangletiming/(Dscalar)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;

//    cout << endl << "GPU time  = " << spv.gputiming/(Dscalar)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;
//    cout << "CPU time  = " << spv.cputiming/(Dscalar)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;

//    ncdat.WriteState(spv);

    return 0;
};
