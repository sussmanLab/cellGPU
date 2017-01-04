#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "avm2d.h"
#include "cu_functions.h"
#include "DatabaseAVM.h"

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
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
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
            case 'd': Dr = atof(optarg); break;
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
    sprintf(dataname,"../test.nc");
    int Nvert = 2*numpts;
    AVMDatabase ncdat(Nvert,dataname,NcFile::Replace);


    AVM2D avm(numpts,1.0,p0,reproducible,initializeGPU);
    if(USE_GPU < 0)
        avm.setCPU();
    avm.setv0Dr(v0,Dr);
    avm.setDeltaT(dt);

    for (int timestep = 0; timestep < initSteps; ++timestep)
        {
        avm.performTimestepGPU();
        if(program_switch <0 && timestep%((int)(1/dt))==0)
            {
            cout << timestep << endl;
            ncdat.WriteState(avm);
            };
        };

    t1=clock();
    if(initializeGPU)
        cudaProfilerStart();
    for (int timestep = 0; timestep < tSteps; ++timestep)
        {
        avm.performTimestep();
        };
    if(initializeGPU)
        cudaProfilerStop();
    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;

//    avm.reportMeanForce();

    if(initializeGPU)
        cudaDeviceReset();

    return 0;
    };

