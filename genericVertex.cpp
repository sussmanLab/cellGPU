#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexModelGenericBase.h"
//#include "vertexQuadraticEnergy.h"
//#include "selfPropelledCellVertexDynamics.h"
//#include "brownianParticleDynamics.h"
//#include "DatabaseNetCDFAVM.h"


void saveConfig(ofstream &output, shared_ptr<vertexModelGenericBase> modelBase)
    {
    ArrayHandle<int> vnn(modelBase->vertexNeighborNum);
    Index2D vni = modelBase->vertexNeighborIndexer;
    ArrayHandle<Dscalar2> pos(modelBase->vertexPositions);
    ArrayHandle<int> vn(modelBase->vertexNeighbors);
    ArrayHandle<int> vcn(modelBase->vertexCellNeighbors);
    int Nv = modelBase->Nvertices;
    int Nc = modelBase->Ncells;
    output << Nv << "\n";
    //write the verte coordinates
    for (int vv = 0; vv < Nv; ++vv)
        {
        output << pos.data[vv].x <<"\t" <<pos.data[vv].y << "\n";
        };
    vector<int2> vertexVertexConnections;
    for (int vv = 0; vv < Nv; ++vv)
        {
        int neighs = vnn.data[vv];
        for (int n = 0; n < neighs; ++n)
            {
            int vIdx = vn.data[vni(n,vv)];
            if (vv < vIdx)
                {
                int2 vvc; vvc.x=vv; vvc.y=vIdx;
                vertexVertexConnections.push_back(vvc);
                }
            };
        };
    output << vertexVertexConnections.size() << "\n";
    for (int vv = 0; vv < vertexVertexConnections.size(); ++vv)
        output << vertexVertexConnections[vv].x << "\t" << vertexVertexConnections[vv].y << "\n";
    };

/*!
This file is for building and testing a more generic version of 2D vertex models, capable of having
vertices of arbitrary coordination number.
*/
int main(int argc, char*argv[])
{
    int numpts = 200; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 5; //number of time steps to run after initialization
    int initSteps = 3; //number of initialization steps

    Dscalar dt = 0.01; //the time step size
    Dscalar p0 = 4.0;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 0.1;  // the self-propulsion
    Dscalar Dr = 1.0;  //the rotational diffusion constant of the cell directors
    int program_switch = 0; //various settings control output

    int c;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
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
    //clocks for timing information
    clock_t t1,t2;

    // if you want random numbers with a more random seed each run, set this to false
    bool reproducible = true;
    //check to see if we should run on a GPU
    bool initializeGPU = setCudaDevice(USE_GPU);

    shared_ptr<vertexModelGenericBase> modelBase = make_shared<vertexModelGenericBase>();
    modelBase->setGPU(initializeGPU);
    modelBase->setReproducible(reproducible);
    modelBase->initializeVertexGenericModelBase(numpts);

    //possibly save output in netCDF format
    char dataname[256];
    sprintf(dataname,"../test.txt");

    modelBase->computeGeometryCPU();
    modelBase->getCellPositions();
    vector<int> cellsToRemove;
    {
    ArrayHandle<Dscalar2> pos(modelBase->cellPositions);
    Dscalar2 center = make_Dscalar2(sqrt(numpts)*0.5,sqrt(numpts)*0.5);
    for (int n =0; n < numpts; ++n)
        {
        Dscalar2 dist;
        modelBase->Box->minDist(pos.data[n],center,dist);
        if(norm(dist) > 3.5)
            cellsToRemove.push_back(n);
        };
    };

    t1=clock();
    modelBase->removeCells(cellsToRemove);
    t2=clock();
    printf("cell removal time: %f\n",(t2-t1)/(Dscalar)CLOCKS_PER_SEC);

    ofstream output(dataname);

    saveConfig(output,modelBase);

/*
    modelBase->computeGeometryCPU();
    ArrayHandle<Dscalar2> ap(modelBase->returnAreaPeri());
    int Nc = modelBase->Ncells;
    for (int ii = 0; ii < Nc; ++ii)
        {
        if(true)
            {
            modelBase->printCellGeometry(ii);
            printf("\n");
            }
        };
*/
/*
    //possibly save output in netCDF format
    char dataname[256];
    sprintf(dataname,"../test.nc");
    int Nvert = 2*numpts;
    AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);

    //We will define two potential equations of motion, and choose which later on.
    //define an equation of motion object...here for self-propelled cells
    EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
    //the next lines declare a potential brownian dynamics scheme at some targe temperature
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
    bd->setT(v0);
    //define a vertex model configuration with a quadratic energy functional
    shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,4.0,reproducible);
    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    avm->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    avm->setv0Dr(v0,1.0);
    //when an edge gets less than this long, perform a simple T1 transition
    avm->setT1Threshold(0.04);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(bd,avm);
    //one could have written "sim->addUpdater(spp,avm);" to use the active cell dynamics instead

    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
//    sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    //perform some initial time steps. If program_switch < 0, save periodically to a netCDF database
    for (int timestep = 0; timestep < initSteps+1; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            //ncdat.WriteState(avm);
            };
        };

    //run for additional timesteps, and record timing information. Save frames to a database if desired
    cudaProfilerStart();
    t1=clock();
    for (int timestep = 0; timestep < tSteps; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            ncdat.WriteState(avm);
            };
        };
    cudaProfilerStop();

    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;
    avm->reportMeanVertexForce();
    cout << "Mean q = " << avm->reportq() << endl;
*/
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
    };
