#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
#include "DatabaseNetCDFSPV.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
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
    shared_ptr<SPV2D> SPV = dynamic_pointer_cast<SPV2D>(spv);

    spv->setCellPreferencesUniform(1.0,p0);
    spv->setv0Dr(v0,1.0);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->setEquationOfMotion(spp,spv);
    sim->setIntegrationTimestep(dt);
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    if(!initializeGPU)
        sim->setCPUOperation(true);
    sim->setReproducible(true);
    //initialize parameters

    char dataname[256];
    sprintf(dataname,"../test.nc");
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
    ncdat.WriteState(SPV);


    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv->reportMeanCellForce(false);

    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%100 ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(SPV);
            };
        sim->performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << spv->reportq() << endl;


    ncdat.ReadState(SPV,0,true);
    ncdat.WriteState(SPV);
    if(initializeGPU)
        cudaDeviceReset();

    SPV->computeGeometryCPU();
    Dscalar2 ans;
    ans = SPV->dAidrj(26,25);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dAidrj(26,26);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dAidrj(26,61);
    printf("%g\t%g\n",ans.x,ans.y);

    ans = SPV->dPidrj(26,25);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dPidrj(26,26);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dPidrj(26,61);
    printf("%g\t%g\n",ans.x,ans.y);

/*
    SPV->computeForces();
    GPUArray<Dscalar2> fs;
    fs.resize(numpts);
    SPV->getForces(fs);
    ArrayHandle<Dscalar2> hf(fs);
    printf("Force on cell %i: (%f,%f)\n",26,hf.data[26].x,hf.data[26].y);
*/
    return 0;
};
