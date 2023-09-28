#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "vector_types.h"

#define ENABLE_CUDA

#include "voronoiQuadraticEnergy.h"
#include "Simulation.h"
#include "selfPropelledParticleDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFSPV.h"

/*!
Provides an example of using the NetCDF database class to write snapshots of a simulation of the 2D
Voronoi model. Note that netCDF database functionality is currently only partially implemented for the
AVM, and some refactoring is in the works. In particular, netCDF will be clunky when the cell number
changes
*/
int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int c;
    int tSteps = 5;
    int initSteps = 0;

    double dt = 0.1;
    double Dr = 1.0;
    double p0 = 4.0;
    double a0 = 1.0;
    double v0 = 0.1;

    int program_switch = 0;
    while((c=getopt(argc,argv,"n:g:m:d:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
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
    printf("Initializing a system with N= %i, p0 = %.2f, v0 = %.2f, Dr = %.3f\n",numpts,p0,v0,Dr);
    sprintf(dataname,"../monodisperse_N%i_p%.4f_v%.2f_Dr%.3f.nc",numpts,p0,v0,Dr);
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace,false);

    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(numpts);

    shared_ptr<VoronoiQuadraticEnergy> spv = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

    spv->setCellPreferencesUniform(1.0,p0);
    spv->setv0Dr(v0,Dr);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    if(program_switch == 0)
        sim->addUpdater(spp,spv);
    else
        sim->addUpdater(bd,spv);
    sim->setIntegrationTimestep(dt);
    //sim->setSortPeriod(initSteps/10);
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    //initialize parameters

    bd->setT(v0*v0/2.0*Dr);

    //initialize
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };

    printf("Finished with initialization...running and saving states\n");

    int logSaveIdx = 0;
    int nextSave = 0;
    t1=clock();
    sim->setCurrentTimestep(0);
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii == nextSave)
            {
            printf(" step %i\n",ii);
            ncdat.writeState(spv);
            nextSave = (int)round(pow(pow(10.0,0.05),logSaveIdx));
            while(nextSave == ii)
                {
                logSaveIdx +=1;
                nextSave = (int)round(pow(pow(10.0,0.05),logSaveIdx));
                };

            };
        sim->performTimestep();
        };
    t2=clock();
    ncdat.writeState(spv);

    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;

    return 0;
};
