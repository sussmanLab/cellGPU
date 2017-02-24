#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "spv2d.h"
#include "DatabaseNetCDFSPV.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
for the 2D SPV model found in the "cellGPU" paper, using the following parameters:
i = 20001
t = 5000
e = 0.05
v=0.01
dr = 1.0
p=3.8
along with some other choices of v0 and p0. The SPV timing is sensitive to how often the
triangulation needs to be updated, so these parameters can be quite important. Note the longer
"warm up" time (i=20001), representing a large number of time steps for the system to move from a
random configuration of cells to one that looks more like a tissue before starting to time the
program.
*/
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

    bool reproducible = true;
    bool initializeGPU = true;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
    else
        initializeGPU = false;

    char dataname[256];
    sprintf(dataname,"/hdd2/data/spv/test.nc");
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);

    SPV2D spv(numpts,1.0,p0,reproducible,initializeGPU);
    if (!initializeGPU)
        spv.setCPU(false);

    spv.setCellPreferencesUniform(1.0,p0);
    spv.setv0Dr(v0,1.0);
    spv.setDeltaT(dt);
    printf("starting initialization\n");
    spv.setSortPeriod(initSteps/10);
    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv.performTimestep();
        };

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv.reportMeanCellForce(false);
    spv.repPerFrame = 0.0;

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
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl << spv.skippedFrames << " skipped frames" << endl << endl;

    if(initializeGPU)
        cudaProfilerStart();

    if(initializeGPU)
        cudaProfilerStop();

//    ncdat.WriteState(spv);
    if(initializeGPU)
        cudaDeviceReset();

ofstream outfile;
outfile.open("../timingSPV.txt",std::ios_base::app);
outfile << numpts <<"\t" << (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << "\n";

    return 0;
};
