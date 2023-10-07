#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "NoseHooverChainNVT.h"
#include "DatabaseNetCDFSPV.h"
#include "vectorValueDatabase.h"
#include "MullerPlatheShear.h"

/*!
This file explores integrating the Nose-Hoover equations of motion
*/
int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 200; //number of cells
    int USE_GPU = 0; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 5; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps

    double dt = 0.01; //the time step size
    double p0 = 3.8;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.1;  // the self-propulsion
    int Nchain = 4;     //The number of thermostats to chain together

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 'm': Nchain = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
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

    clock_t t1,t2; //clocks for timing information
    bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
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
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace,false);

    //define the equation of motion to be used
    shared_ptr<NoseHooverChainNVT> nvt = make_shared<NoseHooverChainNVT>(numpts,Nchain);
    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> vm  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,p0,reproducible);
    //set the temperature and the initial velocities to the desired value
    vm->setCellVelocitiesMaxwellBoltzmann(v0);
    nvt->setT(v0);

    double boxL = sqrt(numpts);
    shared_ptr<MullerPlatheShear> mullerPlathe = make_shared<MullerPlatheShear>(floor(.3/dt),floor(boxL),boxL);
    char dataname2[256];
    sprintf(dataname2,"../testMPprofile.nc");
    vectorValueDatabase vvdat(mullerPlathe->Nslabs,dataname2,NcFile::Replace);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(vm);
    sim->addUpdater(nvt,vm);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };

    double instantaneousTemperature = vm->computeKineticEnergy()/numpts;
    cout << "Target temperature = " << v0 << "; instantaneous temperature after initialization = " << instantaneousTemperature << endl;
    sim->addUpdater(mullerPlathe,vm);
    printf("Finished with initialization..adding a Muller-Plathe updater\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    cout << "current q = " << vm->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    vm->reportMeanCellForce(false);

    //run for additional timesteps, and record timing information
    t1=clock();
    double meanT = 0.0;
    double Tsample = (1/dt);
    for(int ii = 0; ii < tSteps; ++ii)
        {
        ArrayHandle<double> h_kes(nvt->kineticEnergyScaleFactor);
        meanT += h_kes.data[0]/(numpts);
        if(ii%(int)(Tsample) ==0)
            {
            double DeltaP = mullerPlathe->getMomentumTransferred();
            printf("timestep %i\t\t energy %f \t T %f DeltaP %f \n",ii,vm->computeEnergy(),h_kes.data[0]/(numpts),DeltaP);
            ncdat.writeState(vm);
            vector<double> Vprofile;
            mullerPlathe->getVelocityProfile(Vprofile);
            vvdat.writeState(Vprofile,DeltaP/(2.0*(dt*Tsample)*boxL));
            };
        sim->performTimestep();
        };
    t2=clock();
    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << vm->reportq() << endl;
    cout << "<T> = " << meanT / tSteps << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
