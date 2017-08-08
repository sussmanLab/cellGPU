#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "avm2d.h"
#include "selfPropelledCellVertexDynamics.h"
#include "DatabaseNetCDFAVM.h"
/*!
This file compiles to produce an executable that can be used to reproduce the timing information
for the 2D AVM model found in the "cellGPU" paper, using the following parameters:
i = 1000
t = 4000
e = 0.01
dr = 1.0,
along with a range of v0 and p0. This model is not very sensitive to these other choices
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
    AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);

    bool runSPV = false;

    EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
    ForcePtr avm = make_shared<AVM2D>(numpts,1.0,4.0,reproducible,runSPV);
    avm->setCellPreferencesUniform(1.0,p0);
    avm->setv0Dr(v0,1.0);

    shared_ptr<AVM2D> AVM = dynamic_pointer_cast<AVM2D>(avm);
    AVM->setT1Threshold(0.04);


    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->setEquationOfMotion(spp,avm);
    sim->setIntegrationTimestep(dt);
//    sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    for (int timestep = 0; timestep < initSteps+1; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(1/dt))==0)
            {
    //        cout << timestep << endl;
    //        avm.reportAP();
    //        avm.reportMeanVertexForce();
            };
        if(program_switch <0 && timestep%((int)(1/dt))==0)
            {
            cout << timestep << endl;
            ncdat.WriteState(AVM);
            };
        };
    AVM->reportMeanVertexForce(true);
    t1=clock();
    if(initializeGPU)
        cudaProfilerStart();
    for (int timestep = 0; timestep < tSteps; ++timestep)
        {
        sim->performTimestep();
        };
    if(initializeGPU)
        cudaProfilerStop();
    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;

    //avm->reportMeanVertexForce();
    cout << avm->reportq() << endl;

    //For debugging...output the force on every vertex
    if(program_switch <-5)
        {
        ncdat.WriteState(AVM);
        avm->computeForces();
        Dscalar Ei = avm->quadraticEnergy();
        ArrayHandle<Dscalar2> vf(avm->vertexForces);
        Dscalar delX = 0.00000001;
        Dscalar2 zero; zero.x=0.0;zero.y=0.0;
        Dscalar2 dx; dx.x=delX; dx.y = 0.0;
        Dscalar2 mdx; mdx.x=-delX; mdx.y = 0.0;
        Dscalar2 dy; dy.x=0.0; dy.y = delX;
        Dscalar2 mdy; mdy.y=-delX; mdy.x = 0.0;
        for (int ii = 0; ii < Nvert; ++ii)
            {
            GPUArray<Dscalar2> disps;
            disps.resize(Nvert);
            {
            ArrayHandle<Dscalar2> hd(disps);
            for (int ii = 0; ii < Nvert; ++ii) hd.data[ii]=zero;
            };
                {
                ArrayHandle<Dscalar2> hd(disps);
                hd.data[ii]=dx;
                }
            avm->moveDegreesOfFreedom(disps);
            AVMparams->computeGeometryCPU();
            Dscalar fxNumerical=(Ei-avm->quadraticEnergy())/delX;
                {
                ArrayHandle<Dscalar2> hd(disps);
                hd.data[ii]=mdx;
                }
            avm->moveDegreesOfFreedom(disps);
            AVMparams->computeGeometryCPU();
            Dscalar fxdiff = fxNumerical - vf.data[ii].x;
            if (abs(fxdiff) > 1e-5) 
                printf("Fx error: %i\t %f \n",ii,fxdiff);
            //cout << fxNumerical <<"   " << vf.data[ii].x  << "  "<< vf.data[ii].y << endl;
            };
        for (int ii = 0; ii < Nvert; ++ii)
            {
            GPUArray<Dscalar2> disps;
            disps.resize(Nvert);
            {
            ArrayHandle<Dscalar2> hd(disps);
            for (int ii = 0; ii < Nvert; ++ii) hd.data[ii]=zero;
            };
                {
                ArrayHandle<Dscalar2> hd(disps);
                hd.data[ii]=dy;
                }
            avm->moveDegreesOfFreedom(disps);
            AVMparams->computeGeometryCPU();
            Dscalar fyNumerical=(Ei-avm->quadraticEnergy())/delX;
                {
                ArrayHandle<Dscalar2> hd(disps);
                hd.data[ii]=mdy;
                }
            avm->moveDegreesOfFreedom(disps);
            AVMparams->computeGeometryCPU();
            Dscalar fydiff = fyNumerical - vf.data[ii].y;
            if (abs(fydiff) > 1e-6) 
                printf("Fy error: %i\t %g \n",ii,fydiff);
            //cout << fxNumerical <<"   " << vf.data[ii].x  << "  "<< vf.data[ii].y << endl;
            };
        };

    if(initializeGPU)
        cudaDeviceReset();

    return 0;
    };
