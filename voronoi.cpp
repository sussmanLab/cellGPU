#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
//#include "DatabaseNetCDFSPV.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
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

    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    ForcePtr spv = make_shared<SPV2D>(numpts,1.0,4.0,reproducible);

//    selfPropelledParticleDynamics spp(numpts);
    spp->setReproducible(reproducible);
//    SPV2D spv(numpts,1.0,p0,reproducible);
    spv->setEquationOfMotion(spp);

    Simulation sim;
    sim.setEquationOfMotion(spp);
    sim.setConfiguration(spv);
    sim.setIntegrationTimestep(dt);
    //set appropriate CPU and GPU flags
    if(!initializeGPU)
        {
        spp->setCPU();
        spv->setCPU();
        //spv->setCPU(false);
        }
    else
        {
        spp->initializeRNGs(1337,0);
        };
    //initialize parameters
    spp->setReproducible(true);
    spp->setDeltaT(dt);
    spv->setCellPreferencesUniform(1.0,p0);
    spv->setv0Dr(v0,1.0);
    spv->setDeltaT(dt);

//    char dataname[256];
//    sprintf(dataname,"/hdd2/data/spv/test.nc");
//    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);


    printf("starting initialization\n");
    spv->setSortPeriod(initSteps/10);
    for(int ii = 0; ii < initSteps; ++ii)
        {
        spv->performTimestep();
        };

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv->reportMeanCellForce(false);

    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10000 ==0)
            {
            printf("timestep %i\n",ii);
//    ncdat.WriteState(spv);
            };
        spv->performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << spv->reportq() << endl;
    if(initializeGPU)
        cudaProfilerStart();

    if(initializeGPU)
        cudaProfilerStop();

//    ncdat.WriteState(spv);
    if(initializeGPU)
        cudaDeviceReset();

    return 0;
};
