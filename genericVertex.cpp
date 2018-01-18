#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexModelGenericBase.h"
#include "vertexGenericQuadraticEnergy.h"
#include "brownianParticleDynamics.h"


void saveConfig(ofstream &output, shared_ptr<vertexModelGenericBase> modelBase)
    {
    ArrayHandle<int> vnn(modelBase->vertexNeighborNum);
    Index2D vni = modelBase->vertexNeighborIndexer;
    ArrayHandle<Dscalar2> pos(modelBase->vertexPositions);
    ArrayHandle<int> vn(modelBase->vertexNeighbors);
    int Nv = modelBase->Nvertices;
    int Nc = modelBase->Ncells;
    output << Nv << "\n";
    //write the vertex coordinates
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
    int initSteps = 100; //number of initialization steps

    Dscalar dt = 0.001; //the time step size
    Dscalar p0 = 4.0;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar T = 0.01;  // the temperature
    Dscalar Dr = 1.0;  //the rotational diffusion constant of the cell directors
    int program_switch = 0; //various settings control output

    int c;
    while((c=getopt(argc,argv,"n:g:m:s:t:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 't': T = atof(optarg); break;
            case 'e': dt = atof(optarg); break;
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
    //save output
    char dataname[256];
    sprintf(dataname,"../test.txt");
    int sparsity = 50;
    ofstream output(dataname);

    //clocks for timing information
    clock_t t1,t2;

    // if you want random numbers with a more random seed each run, set this to false
    bool reproducible = true;
    //check to see if we should run on a GPU
    bool initializeGPU = setCudaDevice(USE_GPU);

    shared_ptr<VertexGenericQuadraticEnergy> modelBase = make_shared<VertexGenericQuadraticEnergy>(numpts,reproducible);
    modelBase->setGPU(initializeGPU);
    modelBase->computeGeometryCPU();
    saveConfig(output,modelBase);

    int Nvert = 2*numpts;
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
    bd->setT(T);


    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(modelBase);
    sim->addUpdater(bd,modelBase);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    for (int timestep = 0; timestep < initSteps; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(initSteps/sparsity))==0)
            {
            cout << timestep << endl;
            saveConfig(output,modelBase);
            };
        };

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
    modelBase->computeGeometry();
    printf("cell removal time: %f\n",(t2-t1)/(Dscalar)CLOCKS_PER_SEC);


    for (int timestep = 0; timestep < initSteps; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(initSteps/sparsity))==0)
            {
            cout << timestep << endl;
            saveConfig(output,modelBase);
            };
        };



    vector<int> vMerge(2); vMerge[0]=80;vMerge[1]=3;
    t1=clock();
    modelBase->mergeVertices(vMerge);
    t2=clock();
    modelBase->computeGeometry();
    printf("vertex merging time: %f\n",(t2-t1)/(Dscalar)CLOCKS_PER_SEC);

    for (int timestep = 0; timestep < initSteps; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(initSteps/sparsity))==0)
            {
            cout << timestep << endl;
            saveConfig(output,modelBase);
            };
        };

    int NcNew = modelBase->Ncells;
    vector<Dscalar2> prefs(NcNew,make_Dscalar2(1.0,3.8));
    int cellToDie = 30;
    prefs[cellToDie].x=0.1;
    prefs[cellToDie].x=1.1;
    modelBase->setCellPreferences(prefs);
    for (int timestep = 0; timestep < initSteps; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(initSteps/sparsity))==0)
            {
            cout << timestep << endl;
            saveConfig(output,modelBase);
            };
        };
    modelBase->cellDeath(cellToDie);
    modelBase->computeGeometryCPU();

    for (int timestep = 0; timestep < 2*initSteps; ++timestep)
        {
        sim->performTimestep();
        if(timestep%((int)(initSteps/sparsity))==0)
            {
            cout << timestep << endl;
            saveConfig(output,modelBase);
            };
        };
    /*
    ArrayHandle<Dscalar2> ap(modelBase->returnAreaPeri());
    int Nc = modelBase->Ncells;
    for (int ii = 0; ii < Nc; ++ii)
        {
        if(true)
            {
            printf("cell %i: ",ii);
            modelBase->printCellGeometry(ii);
            printf("\n");
            }
        };
    */
/*
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
//    sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags


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
