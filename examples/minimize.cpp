#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "selfPropelledCellVertexDynamics.h"
#include "vertexQuadraticEnergy.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"

/*!
This file compiles to produce an executable that demonstrates how to use the energy minimization
functionality of cellGPU. Now that energy minimization behaves like any other equation of motion, this
demonstration is pretty straightforward
*/

//! A function of convenience for setting FIRE parameters
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
    //as in the examples in the main directory, there are a bunch of default parameters that
    //can be changed from the command line
    int numpts = 200;
    int USE_GPU = 0;
    int c;
    int tSteps = 1000;
    int initSteps = 1000;

    Dscalar dt = 0.1;
    Dscalar KA = 1.0;
    Dscalar p0 = 3.8;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;

    int program_switch = 0;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:k:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'k': KA = atof(optarg); break;
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
    bool reproducible = false;
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

    //program_switch == 0 --> voronoi model
    if(program_switch == 0)
        {
        //initialize parameters and set up simulation
        shared_ptr<VoronoiQuadraticEnergy> spv = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

        shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(spv);

        spv->setCellPreferencesUniform(1.0,p0);
        spv->setModuliUniform(KA,1.0);
        spv->setv0Dr(v0,1.0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(spv);
        sim->addUpdater(fireMinimizer,spv);
        sim->setIntegrationTimestep(dt);
        //if(initSteps > 0)
            //sim->setSortPeriod(initSteps/10);
        sim->setCPUOperation(!initializeGPU);

        SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
        ncdat.WriteState(spv);

        for (int i = 0; i <tSteps;++i)
            {
            //these fire parameters are reasonably standard...
            setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
            //...but incrementing by "50" timesteps here may be *very* short. May require many such loops to be well-minimized
            fireMinimizer->setMaximumIterations(50*(i+1));
            sim->performTimestep();
            ncdat.WriteState(spv);
            };
        printf("minimized value of q = %f\n",spv->reportq());
        ncdat.WriteState(spv);
        };

    //program_switch == 1 --> vertex model
    if(program_switch == 1)
        {
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,p0,reproducible);

        shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(avm);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->addUpdater(fireMinimizer,avm);
        sim->setIntegrationTimestep(dt);
//        if(initSteps > 0)
//           sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
        sim->setCPUOperation(!initializeGPU);

        AVMDatabaseNetCDF ncdat(2*numpts,dataname,NcFile::Replace);
        ncdat.WriteState(avm);
        Dscalar mf;
        for (int i = 0; i <initSteps;++i)
            {
            setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
            fireMinimizer->setMaximumIterations(tSteps*(i+1));
            sim->performTimestep();
            mf = fireMinimizer->getMaxForce();
            if (mf < 1e-12)
                    break;
            ncdat.WriteState(avm);
            };
        printf("minimized value of q = %f\n",avm->reportq());
        Dscalar meanQ = avm->reportq();
        Dscalar varQ = avm->reportVarq();
        Dscalar2 variances = avm->reportVarAP();
        printf("current KA = %f\t Cell <q> = %f\t Var(p) = %g\n",KA,meanQ,variances.y);
        ncdat.WriteState(avm);
        };
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
