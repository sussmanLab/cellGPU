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
    int tSteps = 5;
    int initSteps = 0;

    float dt = 0.1;
    float p0 = 4.0;
    float a0 = 1.0;
    float v0 = 0.1;
    float gamma = 0.0;

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



    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) return 0;
    cudaSetDevice(USE_GPU);



    char dataname[256];
    sprintf(dataname,"/hdd2/data/spv/Plates_N%i_p%.2f_v%.2f_g%.2f.nc",numpts,p0,v0,gamma);
    SPVDatabase ncdat(numpts,dataname,NcFile::Replace,true);
    SPV2D spv(numpts,1.0,p0);

    spv.setCellPreferencesUniform(1.0,p0);
    spv.setv0Dr(v0,1.0);
    spv.setDeltaT(dt);


    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };

    printf("Finished with initialization\n");

    printf("Setting cells within the central circle to different type, adding plates...\n");
    spv.setCellTypeEllipse(0.25,1.0);
    spv.setUseTension(true);
    spv.setTension(gamma);
    float boxL = sqrt(numpts);

    float delta = (4.5*boxL/5.5 -3.*boxL/4.0)/(float)tSteps;

    GPUArray<float2> plateMover(numpts);
    vector<int> excl(numpts,0);
    if(true)
        {
        ArrayHandle<float2> h_pm(plateMover,access_location::host,access_mode::overwrite);
        ArrayHandle<float2> h_p(spv.points,access_location::host,access_mode::read);

        float width = 1.5;
        for (int nn = 0; nn < numpts; ++nn)
            {
            float2 pp = h_p.data[nn];
            h_pm.data[nn].x=0.0;
            h_pm.data[nn].y=0.0;
            if (pp.x > 0.1*boxL && pp.x < 0.9*boxL)
                {
                float ypos = fabs(pp.y-4.5*boxL/5.5);
                if (ypos < width)
                    {
                    h_pm.data[nn].y = -delta;
                    excl[nn]=1;
                    }
                ypos = fabs(pp.y-1.*boxL/5.5);
                if (ypos < width)
                    {
                    h_pm.data[nn].y = delta;
                    excl[nn]=1;
                    };
                };
            };
        };
    spv.setExclusions(excl);

    //first, move plates down
    printf("Moving plates down over ~ %.2f tau \n",tSteps*dt);
    int saveRate = floor(100/dt);
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%saveRate ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(spv);
            };
        spv.movePoints(plateMover);
        spv.performTimestep();
        };

    printf("Relaxing system with plates fixed\n");
    for(int ii = 0; ii < 10*tSteps; ++ii)
        {

        if(ii%saveRate ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(spv);
            };
        spv.performTimestep();
        };
    t2=clock();
    
    
    float steptime = (t2-t1)/(dbl)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl;
    cout << "current q = " << spv.reportq() << endl;
    spv.meanForce();

    cout << endl << "force time  = " << spv.forcetiming/(float)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;
    cout << "other time  = " << spv.triangletiming/(float)CLOCKS_PER_SEC/(initSteps+tSteps) << endl;



    return 0;
};
