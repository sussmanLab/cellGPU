#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "spv2d.h"
#include "DatabaseNetCDFSPV.h"
#include "EnergyMinimizerFIRE2D.h"
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
//    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);



    //SPV2DTension spv(numpts,1.0,p0,reproducible,initializeGPU);
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

    if(initializeGPU)
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
    if(initializeGPU)
        cudaProfilerStop();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl << spv.repPerFrame/tSteps*numpts << " particle  edits per frame; " << spv.GlobalFixes << " calls to the global triangulation routine." << endl << spv.skippedFrames << " skipped frames" << endl << endl;


    if(program_switch ==1)
        {
        EnergyMinimizerFIRE<SPV2D> emin(spv);
        emin.minimize();
        };

//    ncdat.WriteState(spv);
    if(initializeGPU)
        cudaDeviceReset();

ofstream outfile;
outfile.open("../timingSPV.txt",std::ios_base::app);
outfile << numpts <<"\t" << (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << "\n";

    return 0;
};
