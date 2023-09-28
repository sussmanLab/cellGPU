#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "vertexQuadraticEnergyWithTension.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"

/*!
This file compiles to produce an executable that can be used to study vertex model cells with explicit line tension terms between cells of different "type."
One should really consider the interaction between the rules for vertex splitting (e.g. T1 transitions,
etc.) and the presence of inhomogeneous line tensions in the vertex model. See, for instance, the
work of Spencer, Jabeen, and Lubensky for how this can stabilize four-fold vertices, which means that
a default scheme of "perform a T1 transition when edges are sufficiently short" is less sensible.
Really, this functionality should be coupled with a generalization of the vertex model that easily
permits stable multifold vertices. Contact DMS if you are interested in working on such an implementation
in the context of cellGPU!
*/
int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int USE_TENSION = 0;
    int c;
    int tSteps = 100;
    int initSteps = 100;

    double dt = 0.01;
    double p0 = 4.0;
    double a0 = 1.0;
    double v0 = 0.01;
    double gamma = 0.0;

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

    //possibly save output in netCDF format
    char dataname[256];
    sprintf(dataname,"../test.nc");
    int Nvert = 2*numpts;
    AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);

    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
    bd->setT(v0);
    //define a vertex model configuration with a quadratic energy functional
    shared_ptr<VertexQuadraticEnergyWithTension> avm = make_shared<VertexQuadraticEnergyWithTension>(numpts,1.0,4.0,reproducible,true);
    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    avm->setCellPreferencesUniform(1.0,p0);
    //when an edge gets less than this long, perform a simple T1 transition
    avm->setT1Threshold(0.04);


    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(bd,avm);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    //perform some initial time steps.
    cout << "starting initialization" << endl;
    for (int timestep = 0; timestep < initSteps+1; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(1000/dt))==0)
            {
            cout << timestep << endl;
            ncdat.writeState(avm);
            };
        };

    //set a strip in the middle of the box?
    cout << "Setting a strip with tension = " << gamma << endl;
    avm->getCellPositionsCPU();
    ArrayHandle<double2> h_cp(avm->cellPositions);
    vector<int> types(numpts,0);
    double boxL = (double) sqrt(numpts);
    for (int ii = 0; ii < numpts; ++ii)
        {
        if (h_cp.data[ii].x > boxL/4. && h_cp.data[ii].x < 3.*boxL/4.) types[ii]=1;
        }
    avm->setCellType(types);
    avm->setSurfaceTension(gamma);
    avm->setUseSurfaceTension(true);


    //run for additional timesteps, and record timing information. Save frames to a database if desired
    t1=clock();
    cout << "running and saving states" << endl;
    for (int timestep = 0; timestep < tSteps; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(1000/dt))==0)
            {
            cout << timestep << endl;
            ncdat.writeState(avm);
            };
        };

    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(double)CLOCKS_PER_SEC/tSteps << endl << endl;
    avm->reportMeanVertexForce();
    cout << "Mean q = " << avm->reportq() << endl;


    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
