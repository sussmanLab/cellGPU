#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "avm2d.h"
#include "noiseSource.h"
#include "voronoiTension2d.h"
#include "selfPropelledCellVertexDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseTextVoronoi.h"
/*!
This file demonstrates simulations in the vertex or voronoi models in which a cell divides.
The vertex model version (accessed by using a negative "-z" option on the command line) does cell
division in a homogenous, simple vertex model setting. The voronoi model version ("-z" >=0) uses a
two-type model with a very weak tension between the (otherwise identical) cell types. In both cases,
cells are chosen to divide at random.
*/
int main(int argc, char*argv[])
{
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
    noiseSource noise;
    noise.Reproducible = reproducible;

    //program_switch >= 0 --> self-propelled voronoi model
    if(program_switch >=0)
        {
        DatabaseTextVoronoi db1("../test1.txt",0);
        EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
        shared_ptr<VoronoiTension2D> spv = make_shared<VoronoiTension2D>(numpts,1.0,4.0,reproducible);

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
        sim->setEquationOfMotion(spp,spv);
        sim->setIntegrationTimestep(dt);
        sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
        sim->setCPUOperation(!initializeGPU);
        sim->setReproducible(reproducible);


        for (int timestep = 0; timestep < initSteps+1; ++timestep)
            {
            sim->performTimestep();
            if(program_switch == 2 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                db1.WriteState(spv);
                };
            };

        vector<int> cdtest(1); cdtest[0]=10;
        vector<Dscalar> dParams(2); dParams[0] = 3.0*PI/4.0-.1; dParams[1] = 0.5;
        int Ncells = spv->getNumberOfDegreesOfFreedom();

        char dataname2[256];
        int divisionTime = 20;
        t1=clock();
        int fileidx=1;
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
                Dscalar meanA = numpts / (Dscalar) Ncells;
                Dscalar scaledP0 = p0 * sqrt(meanA); 
                spv->setCellPreferencesUniform(1.0,scaledP0);
                printf("Ncells = %i\t <A> = %f \t p0 = %f\n",Ncells,meanA,scaledP0);
                };
            if(program_switch == 2 && timestep%((int)(10/dt))==0)
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
        int Nvert = 2*numpts;
        AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
        bool runSPV = false;

        EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
        shared_ptr<AVM2D> avm = make_shared<AVM2D>(numpts,1.0,4.0,reproducible,runSPV);
        avm->setCellPreferencesUniform(1.0,p0);
        avm->setv0Dr(v0,1.0);

        avm->setT1Threshold(0.04);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->setEquationOfMotion(spp,avm);
        sim->setIntegrationTimestep(dt);
        sim->setSortPeriod(initSteps/10);
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
            if(program_switch < -1 && timestep%((int)(1/dt))==0)
                {
                cout << timestep << endl;
                ncdat.WriteState(avm);
                };
            };
        vector<int> cdtest(3); cdtest[0]=10; cdtest[1] = 0; cdtest[2] = 2;
        avm->cellDivision(cdtest);

        t1=clock();
        int Nvertices = avm->getNumberOfDegreesOfFreedom();
        int Ncells = Nvertices/2;
        int fileidx=2;
        for (int timestep = 0; timestep < tSteps; ++timestep)
            {
            sim->performTimestep();
            if(program_switch <=-1 && timestep%((int)(10/dt))==0)
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
                ncdat2.WriteState(avm);
                };
            };

        t2=clock();
        cout << "final number of vertices = " <<avm->getNumberOfDegreesOfFreedom() << endl;
        cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;

        avm->reportMeanVertexForce();
        cout << "Mean q = " << avm->reportq() << endl;
        };


    if(initializeGPU)
        cudaDeviceReset();

    return 0;
    };
