#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexQuadraticEnergy.h"
#include "noiseSource.h"
#include "voronoiQuadraticEnergyWithTension.h"
#include "selfPropelledCellVertexDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseTextVoronoi.h"
#include "periodicBoundaries.h"

/*!
This file demonstrates simulations in the vertex or voronoi models in which a cell divides.
The vertex model version (accessed by using a negative "-z" option on the command line) does cell
division in a homogenous, simple vertex model setting. The voronoi model version ("-z" >=0) uses a
two-type model with a very weak tension between the (otherwise identical) cell types. In both cases,
cells are chosen to divide at random.
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

    double dt = 0.01;
    double p0 = 3.84;
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
        EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
        shared_ptr<VoronoiQuadraticEnergyWithTension> spv = make_shared<VoronoiQuadraticEnergyWithTension>(numpts,1.0,4.0,reproducible);

        //for variety, we'll have cell division between two types of cells, with some applied surface tension between the types
        //...this section does the usual business of setting up the simulation
        vector<int> types(numpts);
        for (int ii = 0; ii < numpts; ++ii)
            if (ii < numpts/2)
                types[ii]=0;
            else
                types[ii]=1;
        spv->setCellType(types);
        spv->setSurfaceTension(gamma);
        spv->setUseSurfaceTension(true);

        spv->setCellPreferencesUniform(1.0,p0);
        spv->setv0Dr(v0,1.0);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(spv);
        sim->addUpdater(spp,spv);
        sim->setIntegrationTimestep(dt);
        //sim->setSortPeriod(initSteps/10);
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

        //to have a Voronoi model division, one needs to pass in various parameters.
        //the integer vector (of length 1) indicates the cell to divide
        //the double vector (of length 2) parameterizes the geometry of the cell division... see voronoiModelBase for details
        vector<int> cdtest(1); cdtest[0]=10;
        vector<double> dParams(2); dParams[0] = 3.0*PI/4.0-.1; dParams[1] = 0.5;
        int Ncells = spv->getNumberOfDegreesOfFreedom();

        //in this example, divide a cell every 20 tau
        int divisionTime = 20;
        t1=clock();
        for (int timestep = 0; timestep < tSteps; ++timestep)
            {
            sim->performTimestep();
            if(program_switch >0 && timestep%((int)(divisionTime/dt))==0)
                {
                cdtest[0] = noise.getInt(0,Ncells-1);
                dParams[0] = noise.getRealUniform(0,PI);
                spv->cellDivision(cdtest,dParams);
                Ncells = spv->getNumberOfDegreesOfFreedom();
                //suppose, for instance, you want to keep p_0/sqrt(<A>) constant...
                double meanA = numpts / (double) Ncells;
                double scaledP0 = p0 * sqrt(meanA);
                spv->setCellPreferencesUniform(1.0,scaledP0);
                printf("Ncells = %i\t <A> = %f \t p0 = %f\n",Ncells,meanA,scaledP0);
/*
 //An alternative would be to use something like the following to rescale the box size to keep <A> = 1, and not rescale the preferred perimeter
PeriodicBoxPtr newbox = make_shared<periodicBoundaries>(sqrt(Ncells),sqrt(Ncells));
sim->setBox(newbox);
*/
                };
            if(program_switch == 2 && timestep%((int)(10/dt))==0)
                {
                cout << timestep << endl;
                db1.writeState(spv);
                };
            };

        t2=clock();
        cout << "timestep time per iteration currently at " <<  (t2-t1)/(double)CLOCKS_PER_SEC/tSteps << endl << endl;

        };

    //program_switch < 0 --> self-propelled vertex model
    if(program_switch <0)
        {
        //...this section does the usual business of setting up the simulation
        int Nvert = 2*numpts;
        //here we'll demonstrate the more awkward task of using a sequence of netCDF databases to record what's going on
        AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
        bool runSPV = false;

        EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,4.0,reproducible,runSPV);
        avm->setCellPreferencesUniform(1.0,p0);
        avm->setv0Dr(v0,1.0);
        avm->setT1Threshold(0.04);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->addUpdater(spp,avm);
        sim->setIntegrationTimestep(dt);
        sim->setSortPeriod(initSteps/10);
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

        //in the vertex model cell division, an integer vector (of length 3) is used.
        //the first indicates the cell to divide, and the next two indicate the vertices at the CCW
        //start of the edge to get new vertices. See vertexModelBase for details
        vector<int> cdtest(3); cdtest[0]=10; cdtest[1] = 0; cdtest[2] = 2;
        avm->cellDivision(cdtest);

        t1=clock();
        int Nvertices = avm->getNumberOfDegreesOfFreedom();
        int Ncells = Nvertices/2;
        int fileidx=2;
        int divisionTime = 10;
        for (int timestep = 0; timestep < tSteps; ++timestep)
            {
            sim->performTimestep();
            if(program_switch <=-1 && timestep%((int)(divisionTime/dt))==0)
                {
                cdtest[0] = noise.getInt(0,Ncells);
                avm->cellDivision(cdtest);

                Nvertices = avm->getNumberOfDegreesOfFreedom();
                Ncells = Nvertices/2;
                };
            if(program_switch == -2 && timestep%((int)(10/dt))==0)
                {
                cout << timestep << endl;
                char dataname2[256];
                sprintf(dataname2,"../test%i.nc",fileidx);
                fileidx +=1;
                AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
                ncdat2.writeState(avm);
                };
            };

        t2=clock();
        cout << "final number of vertices = " <<avm->getNumberOfDegreesOfFreedom() << endl;
        cout << "timestep time per iteration currently at " <<  (t2-t1)/(double)CLOCKS_PER_SEC/tSteps << endl << endl;
        avm->reportMeanVertexForce();
        cout << "Mean q = " << avm->reportq() << endl;
        };

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
    };
