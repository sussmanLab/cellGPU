#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledAligningParticleDynamics.h"
#include "vectorValueDatabase.h"
#include "dynamicalFeatures.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
*/
int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 200; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 5; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps

    double dt = 0.01; //the time step size
    double p0 = 3.8;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.1;  // the self-propulsion
    double Dr  =0.5;  // rotational diffusion
    double J = 0.0;   //alignment coupling
    int fIdx = 0;
    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:j:x:y:z:p:f:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'j': J = atof(optarg); break;
            case 'f': fIdx = atoi(optarg); break; 
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
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
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
    sprintf(dataname,"./data/vvCorr_N%i_p0%.3f_v0%.3f_J%.3f_fidx%i.nc",numpts,p0,v0,J,fIdx);
    char dataname2[256];
    sprintf(dataname2,"./data/Phi_N%i_p0%.3f_v0%.3f_J%.3f_fidx%i.nc",numpts,p0,v0,J,fIdx);

    //define an equation of motion object...here for self-propelled cells
    shared_ptr<selfPropelledAligningParticleDynamics> spp = make_shared<selfPropelledAligningParticleDynamics>(numpts);
    spp->setJ(J);
    cout << "setting the alignment coupling at " << J << endl;
    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> spv  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    spv->setv0Dr(v0,Dr);


    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp,spv);
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
    printf("Finished with initialization\n");
    cout << "current q = " << spv->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    spv->reportMeanCellForce(false);

    //run for additional timesteps, and record timing information
    //Additionally, every tau record the Vicsek order parameter
    int averages = 0;
    double Phi = 0.0;
    double2 vPar, vPerp;
    t1=clock();
    dynamicalFeatures dynFeat(spv->returnPositions(),spv->Box);
    vectorValueDatabase vvdat(3,dataname2,NcFile::Replace);   
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%((int)(10.0/dt))==0)
            {
            vector<double> saveVec(3);
            averages +=1;
            double val = spv->vicsekOrderParameter(vPar, vPerp);
            saveVec[0] = val;
            saveVec[1] = vPar.x;
            saveVec[2] = vPar.y;
            vvdat.writeState(saveVec,10.0/dt);
            Phi += val;
            printf("timestep %i\t\t energy %f\t\t phi %f \n",ii,spv->computeEnergy(),val);
            };
        sim->performTimestep();
        };
    Phi /= averages;
    t2=clock();

    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << spv->reportq() << endl;
    printf("<Phi> = %f\n",Phi);

    //get the v-v spatial correlation function
    double val = spv->vicsekOrderParameter(vPar, vPerp);
    double L = sqrt(numpts);
    double binWidth = 0.5;
    int totalBins = floor(0.5*L/binWidth);

    //x-component of each will be parallel, y-component will be perp.
    vector<double> vvCorr(totalBins);
    vector<double> perBin(totalBins);
    double2 disp;
    ArrayHandle<double2> points(spv->returnPositions());
    ArrayHandle<double2> vel(spv->returnVelocities());
    //loop through points
    for (int ii = 0; ii < numpts - 1; ++ii)
        {
        for (int jj = ii+1;jj < numpts; ++jj)
            {
            spv->Box->minDist(points.data[ii],points.data[jj],disp);
            int iBin = floor(norm(disp)/binWidth);
            if(iBin < totalBins)
                {
                double2 v1 = vel.data[ii];
                double2 v2 = vel.data[jj];
                perBin[iBin] += 1.0;
                vvCorr[iBin] += dot(v1,v2) / dot(v1,v1);
                }
            /*
            int parBin = floor(fabs(dot(disp,vPar))/binWidth);
            int perpBin = floor(fabs(dot(disp,vPerp))/binWidth);
            if(parBin < totalBins)
                {
                vvCorr[parBin].x +=0;
                perBin[parBin].x +=1.0;
                }
            if(perpBin < totalBins)
                {
                vvCorr[perpBin].y +=0;
                perBin[perpBin].y +=1.0;
                }
            */
            };
        };

    for (int bb = 0; bb < totalBins; ++bb)
        {
        if(perBin[bb] >0)
            {
            vvCorr[bb] /= perBin[bb];
            };
        };
    vectorValueDatabase vvdatVV(totalBins,dataname,NcFile::Replace);
    vvdatVV.writeState(vvCorr,binWidth);

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
