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

#include "spv2d.h"
#include "cu_functions.h"
#include "Database.h"


using namespace std;


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
    char dataname[256];
    sprintf(dataname,"/hdd2/data/spv/test.nc");
//    SPVDatabase ncdat(numpts,dataname,NcFile::Replace);



    SPV2D spv(numpts,1.0,p0,true);
    if (USE_GPU < 0)
        spv.setCPU(false);

    spv.setCellPreferencesUniform(1.0,p0);
    spv.setv0Dr(v0,1.0);
    spv.setDeltaT(dt);

    if(program_switch == -3)
        {
        sprintf(dataname,"/hdd2/data/spv/Plates/Plates_N5000_p4.000_v0.100_Dr1.000_g0.100.nc");
        SPVDatabase ncdat(numpts,dataname,NcFile::ReadOnly);
        for (int rr = 0; rr <ncdat.GetNumRecs(); ++rr)
            {
            ncdat.ReadState(spv,rr);
            cout << "frame " << rr << "  q= " <<  spv.reportq() << endl;
            };
        return 0;

        };
    if(program_switch == -2)
        {
        sprintf(dataname,"/hdd2/data/spv/MSD/monodisperse_N5000_p3.84_v0.01_Dr1.000.nc");
        SPVDatabase ncdat(numpts,dataname,NcFile::ReadOnly);
        for (int rr = 0; rr <ncdat.GetNumRecs(); ++rr)
            {
            ncdat.ReadState(spv,rr);
            cout << "frame " << rr << "  q= " <<  spv.reportq() << endl;
            };
        return 0;

        };

    if(program_switch == -1)
        {
        //compare with output of mattias' code
        char fn[256];
        //sprintf(fn,"/home/daniel/Dropbox/test.txt");
        sprintf(fn,"/Users/danielsussman/Dropbox/test.txt");
        ifstream input(fn);
        spv.readTriangulation(input);
        spv.globalTriangulationCGAL();
        spv.allDelSets();
        spv.computeGeometryGPU();
        spv.computeSPVForceSetsGPU();
        spv.sumForceSets();
        spv.reportForces();
        };

    //printf("starting initialization\n");
    spv.setSortPeriod(initSteps/10);
    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };

    //printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv.meanForce();
    spv.repPerFrame = 0.0;

    cudaProfilerStart();
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10000 ==0)
            {
            printf("timestep %i\n",ii);
//    ncdat.WriteState(spv);
            };
        spv.performTimestep();
        };
    t2=clock();
    cudaProfilerStop();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl << spv.skippedFrames << " skipped frames" << endl << endl;


//    ncdat.WriteState(spv);
    cudaDeviceReset();
    return 0;
};
