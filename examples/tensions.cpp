#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergyWithTension.h"
#include "selfPropelledParticleDynamics.h"
#include "DatabaseNetCDFSPV.h"

/*!
This file compiles to produce an executable that can be used to study Voronoi cells with explicit line tension terms between cells of different "type"
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
    shared_ptr<VoronoiQuadraticEnergyWithTension> spv = make_shared<VoronoiQuadraticEnergyWithTension>(numpts,1.0,4.0,reproducible);

    spv->setCellPreferencesUniform(1.0,p0);
    spv->setv0Dr(v0,1.0);

    vector<int> types(numpts);
    for (int ii = 0; ii < numpts; ++ii) types[ii]=ii;
    spv->setCellType(types);
    //"setSurfaceTension" with a single Dscalar just declares a single tension value to apply whenever cells of different type are in contact
    spv->setSurfaceTension(gamma);
    spv->setUseSurfaceTension(true);

    //in contrast, setSurfaceTension with a vector allows an entire matrix of type-type interactions to be specified
    //vector<Dscalar> gams(numpts*numpts,gamma);
    //spv->setSurfaceTension(gams);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp,spv);
    sim->setIntegrationTimestep(dt);
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    //initialize parameters

    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };

    printf("Finished with initialization\n");
    cout << "current q = " << spv->reportq() << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
