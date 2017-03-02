#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
#include "selfPropelledCellVertexDynamics.h"
#include "avm2d.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"

/*!
This file compiles to produce an executable that shows how to use the energy minimization
functionality of cellGPU. Note that the choice of CPU or GPU operation for the minimization class
is independent of the choice of CPU or GPU operation of the cell model used.
*/

void setFIREParameters(EnergyMinimizerFIRE &emin, Dscalar deltaT, Dscalar alphaStart,
        Dscalar deltaTMax, Dscalar deltaTInc, Dscalar deltaTDec, Dscalar alphaDec, int nMin,
        Dscalar forceCutoff)
    {
    emin.setDeltaT(deltaT);
    emin.setAlphaStart(alphaStart);
    emin.setDeltaTMax(deltaTMax);
    emin.setDeltaTInc(deltaTInc);
    emin.setDeltaTDec(deltaTDec);
    emin.setAlphaDec(alphaDec);
    emin.setNMin(nMin);
    emin.setForceCutoff(forceCutoff);
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
        SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
        SPV2D spv(numpts,1.0,p0,reproducible);
        spv.setCellPreferencesUniform(1.0,p0);
        spv.setv0Dr(v0,1.0);
        selfPropelledParticleDynamics spp(numpts);
        spp.setDeltaT(dt);
        spv.setEquationOfMotion(spp);
        if (!initializeGPU)
            {
            spv.setCPU(false);
            spp.setCPU();
            }
        else
            spp.initializeRNGs(1337,0);
        printf("starting initialization\n");
        spv.setSortPeriod(initSteps/10);
        for(int ii = 0; ii < initSteps; ++ii)
            {
            spv.performTimestep();
            };
        if(initializeGPU)
            cudaProfilerStart();
        ncdat.WriteState(spv);
        for (int i = 0; i <tSteps;++i)
            {
            EnergyMinimizerFIRE emin(spv);
            setFIREParameters(emin,0.01,0.99,0.1,1.1,0.95,.9,4,1e-12);
            if(USE_GPU >=0 )
                emin.setGPU();
            else
                emin.setCPU();
            emin.setMaximumIterations(50);
            emin.minimize();
            ncdat.WriteState(spv);
            };
        printf("minimized value of q = %f\n",spv.reportq());
        if(initializeGPU)
            cudaProfilerStop();
        ncdat.WriteState(spv);
        if(initializeGPU)
            cudaDeviceReset();
    };
    if(program_switch == 1)
        {
        AVM2D avm(numpts,1.0,p0,reproducible,true);
        AVMDatabaseNetCDF ncdat(avm.Nvertices,dataname,NcFile::Replace);
        selfPropelledCellVertexDynamics sppCV(numpts,2*numpts);
        sppCV.setDeltaT(dt);
        avm.setEquationOfMotion(sppCV);
        if (!initializeGPU)
            {
            avm.setCPU();
            sppCV.setCPU();
            }
        else
            sppCV.initializeRNGs(1337,0);
        avm.setCellPreferencesUniform(1.0,p0);
        avm.setv0Dr(v0,1.0);
        avm.setDeltaT(dt);
        printf("starting initialization\n");
        avm.setSortPeriod(initSteps/10);
        for(int ii = 0; ii < initSteps; ++ii)
            {
            avm.performTimestep();
            };
        if(initializeGPU)
            cudaProfilerStart();
        ncdat.WriteState(avm);
        for (int i = 0; i <tSteps;++i)
            {
            EnergyMinimizerFIRE emin(avm);
            setFIREParameters(emin,0.01,0.99,0.1,1.1,0.95,.9,4,1e-12);
            if(USE_GPU >=0 )
                emin.setGPU();
            else
                emin.setCPU();
            emin.setMaximumIterations(50);
            emin.minimize();
            ncdat.WriteState(avm);
            };
        printf("minimized value of q = %f\n",avm.reportq());
        if(initializeGPU)
            cudaProfilerStop();
        ncdat.WriteState(avm);
        if(initializeGPU)
            cudaDeviceReset();
    };
    return 0;
};
