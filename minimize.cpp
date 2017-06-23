#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
#include "selfPropelledCellVertexDynamics.h"
#include "avm2d.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"

/*!
This file compiles to produce an executable that demonstrates how to use the energy minimization
functionality of cellGPU. Now that energy minimization behaves like any other equation of motion, this
demonstration is pretty straightforward
*/

void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, Dscalar deltaT, Dscalar alphaStart,
        Dscalar deltaTMax, Dscalar deltaTInc, Dscalar deltaTDec, Dscalar alphaDec, int nMin,
        Dscalar forceCutoff)
    {
    emin->setDeltaT(deltaT);
    emin->setAlphaStart(alphaStart);
    emin->setDeltaTMax(deltaTMax);
    emin->setDeltaTInc(deltaTInc);
    emin->setDeltaTDec(deltaTDec);
    emin->setAlphaDec(alphaDec);
    emin->setNMin(nMin);
    emin->setForceCutoff(forceCutoff);
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

    if(program_switch == 0)
        {
        ForcePtr spv = make_shared<SPV2D>(numpts,1.0,4.0,reproducible);
        shared_ptr<SPV2D> SPV = dynamic_pointer_cast<SPV2D>(spv);

        EOMPtr fireMinimizer = make_shared<EnergyMinimizerFIRE>(spv);
        shared_ptr<EnergyMinimizerFIRE> FIREMIN = dynamic_pointer_cast<EnergyMinimizerFIRE>(fireMinimizer);

        spv->setCellPreferencesUniform(1.0,p0);
        spv->setv0Dr(v0,1.0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(spv);
        sim->setEquationOfMotion(fireMinimizer,spv);
        sim->setIntegrationTimestep(dt);
        if(initSteps > 0)
            sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
        sim->setCPUOperation(!initializeGPU);

        SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
        ncdat.WriteState(SPV);

        for (int i = 0; i <tSteps;++i)
            {
            setFIREParameters(FIREMIN,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
            FIREMIN->setMaximumIterations(50*(i+1));
            sim->performTimestep();
            ncdat.WriteState(SPV);
            };
        printf("minimized value of q = %f\n",spv->reportq());
        ncdat.WriteState(SPV);
        };
    if(program_switch == 1)
        {
        ForcePtr avm = make_shared<AVM2D>(numpts,1.0,4.0,reproducible);
        shared_ptr<AVM2D> AVM = dynamic_pointer_cast<AVM2D>(avm);

        EOMPtr fireMinimizer = make_shared<EnergyMinimizerFIRE>(avm);
        shared_ptr<EnergyMinimizerFIRE> FIREMIN = dynamic_pointer_cast<EnergyMinimizerFIRE>(fireMinimizer);

        avm->setCellPreferencesUniform(1.0,p0);
        avm->setv0Dr(v0,1.0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->setEquationOfMotion(fireMinimizer,avm);
        sim->setIntegrationTimestep(dt);
        if(initSteps > 0)
            sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
        sim->setCPUOperation(!initializeGPU);

        AVMDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
        ncdat.WriteState(AVM);

        for (int i = 0; i <tSteps;++i)
            {
            setFIREParameters(FIREMIN,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
            FIREMIN->setMaximumIterations(50*(i+1));
            sim->performTimestep();
            ncdat.WriteState(AVM);
            };
        printf("minimized value of q = %f\n",avm->reportq());
        ncdat.WriteState(AVM);
        };
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
