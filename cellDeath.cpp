#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexQuadraticEnergy.h"
#include "noiseSource.h"
#include "voronoiQuadraticEnergy.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseTextVoronoi.h"
#include "gpubox.h"

/*!
This file demonstrates simulations in the vertex or voronoi models in which a cell dies.
The vertex model version (accessed by using a negative "-z" option on the command line) does cell
division in a homogenous, simple vertex model setting. The voronoi model version is accesed with "-z" >=0.  In both cases, cells are chosen to divide at random.
*/
int main(int argc, char*argv[])
{
    //as in the examples in the main directory, there are a bunch of default parameters that
    //can be changed from the command line
    int numpts = 200;
    int USE_GPU = 0;
    int USE_TENSION = 0;
    int c;
    int tSteps = 5;
    int initSteps = 0;

    Dscalar dt = 0.01;
    Dscalar p0 = 3.84;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.01;
    Dscalar Dr = 1.0;
    Dscalar gamma = 0.05;

    //program_switch plays an awkward role in this example of both selecting vertex vs Voronoi model,
    //and also determining whether to save output files... read below for details
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
    sprintf(dataname,"../test1.nc");

    //noiseSources are random number generators used within the code... here we'll randomly select
    //cells to divide
    noiseSource noise;
    noise.Reproducible = reproducible;

    //program_switch >= 0 --> self-propelled voronoi model
    if(program_switch >=0)
        {
        //netCDF databases require the same number of cells in every frame... the text databases lift that limitation, so are useful here
        DatabaseTextVoronoi db1("../test1.txt",0);
        shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(numpts);
        bd->setT(v0);
        shared_ptr<VoronoiQuadraticEnergy> spv = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

        spv->setCellPreferencesUniform(1.0,p0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(spv);
        sim->addUpdater(bd,spv);
        sim->setIntegrationTimestep(dt);
        sim->setSortPeriod(initSteps/2);
        sim->setCPUOperation(!initializeGPU);
        sim->setReproducible(reproducible);

        //perform some initialization timesteps
        //if program_switch = 2, save output file every so often
        for (int timestep = 0; timestep < initSteps+1; ++timestep)
            {
            sim->performTimestep();
            if(program_switch == 2 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                db1.WriteState(spv);
                };
            };

        //to have a Voronoi model cell death, just pick a current index of a cell to die
        int Ncells = spv->getNumberOfDegreesOfFreedom();

        //in this example, divide a cell every 20 tau
        int divisionTime = 20;
        t1=clock();
        for (int timestep = 0; timestep < tSteps; ++timestep)
            {
            sim->performTimestep();
            if(program_switch >0 && timestep%((int)(divisionTime/dt))==0)
                {
                int deadIdx = noise.getInt(0,Ncells-1);
                printf("killing cell %i\n", deadIdx);
                spv->cellDeath(deadIdx);
                Ncells = spv->getNumberOfDegreesOfFreedom();
                //rescale the box size to sqrt(N) by sqrt(N)
                BoxPtr newbox = make_shared<gpubox>(sqrt(Ncells),sqrt(Ncells));
                sim->setBox(newbox);
                };
            if(program_switch == 2 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                db1.WriteState(spv);
                };
            };

        t2=clock();
        cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;

        };

    //program_switch < 0 --> self-propelled vertex model
    if(program_switch <0)
        {

        };

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
    };
