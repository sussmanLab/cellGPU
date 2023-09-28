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
#include "periodicBoundaries.h"

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

    double dt = 0.005;
    double p0 = 3.85;
    double a0 = 1.0;
    double v0 = 0.01;
    double Dr = 1.0;
    double gamma = 0.05;

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
        //sim->setSortPeriod(initSteps/2);
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
                db1.writeState(spv);
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
                PeriodicBoxPtr newbox = make_shared<periodicBoundaries>(sqrt(Ncells),sqrt(Ncells));
                sim->setBox(newbox);
                };
            if(program_switch == 2 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                db1.writeState(spv);
                };
            };

        t2=clock();
        cout << "timestep time per iteration currently at " <<  (t2-t1)/(double)CLOCKS_PER_SEC/tSteps << endl << endl;

        };

    //program_switch < 0 --> vertex model
    if(program_switch <0)
        {
        //...this section does the usual business of setting up the simulation
        int Nvert = 2*numpts;
        //here we'll demonstrate the more awkward task of using a sequence of netCDF databases to record what's going on
        AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
        bool runSPV = false;

        shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
        bd->setT(v0);
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,4.0,reproducible,runSPV);
        avm->setCellPreferencesUniform(1.0,p0);
        avm->setT1Threshold(0.04);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->addUpdater(bd,avm);
        sim->setIntegrationTimestep(dt);
//        sim->setSortPeriod(initSteps/10);
        sim->setCPUOperation(!initializeGPU);
        sim->setReproducible(reproducible);
        //initial time steps
        for (int timestep = 0; timestep < initSteps+1; ++timestep)
            {
            sim->performTimestep();
            if(program_switch < -1 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                ncdat.writeState(avm);
                };
            };

        //now, time to kill some cells. Our strategy will be to take a cell, have it want zero area and perimeter, and kill it when it's a triangle
        int fileidx=2;
        int divisionTime = 10;
        for (int timestep = 0; timestep < tSteps; ++timestep)
            {
            sim->performTimestep();
            if(program_switch <=-1 && timestep%((int)(divisionTime/dt))==0)
                {
                cout << "starting timestep "<<timestep << endl;
                int Nvertices = avm->getNumberOfDegreesOfFreedom();
                int Ncells = Nvertices/2;
                int deadCell = noise.getInt(0,Ncells);
//                cout << "targeting cell " << deadCell << endl;
                double2 oldAP; oldAP.x=1.; oldAP.y = p0;
                vector<double2> newPrefs(Ncells,oldAP);
                newPrefs[deadCell].x = 0.0;
                newPrefs[deadCell].y = p0*0.1;
                avm->setCellPreferences(newPrefs);
                int cellVertices = 0;
                //run for up to one tau to see if the cell shrinks to a triangle...if not, restore
                //the area and perimeter preference to stop the simulation from becoming unstable
                for (int tt =0; tt < 1/dt; ++tt)
                    {
                        {
                        ArrayHandle<int> cn(avm->cellVertexNum,access_location::host,access_mode::read);
                        cellVertices = cn.data[deadCell];
                        };
                    if(cellVertices==3) break;
                    sim->performTimestep();
                    };
                if(cellVertices ==3)
                    {
                    if(program_switch ==-2)
                        {
                        char dataname2[256];
                        sprintf(dataname2,"../test%i.nc",fileidx);
                        fileidx +=1;
                        AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
                        ncdat2.writeState(avm);
                        };
                    cout << "killing cell " << deadCell << endl;
                    avm->cellDeath(deadCell);
                    }
                else
                    {//if the cell doesn't die, restore it's area preferences
                    double2 oldAP; oldAP.x=1.; oldAP.y = p0;
                    vector<double2> newPrefs(Ncells,oldAP);
                    avm->setCellPreferences(newPrefs);
                    };
                };
            if(program_switch <=-2 && (timestep%((int)(0.2*divisionTime/dt)))==0)
                {
                char dataname2[256];
                sprintf(dataname2,"../test%i.nc",fileidx);
                fileidx +=1;
                AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
                ncdat2.writeState(avm);
                };

            };
            char dataname2[256];
            sprintf(dataname2,"../test%i.nc",fileidx);
            AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
            ncdat2.writeState(avm);

        };

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
    };
