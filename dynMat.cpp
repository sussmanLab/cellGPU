#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
#include "EnergyMinimizerFIRE2D.h"
#include "DatabaseNetCDFSPV.h"
#include "eigenMatrixInterface.h"

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
*/

//! A function of convenience for setting FIRE parameters
void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, Dscalar deltaT, Dscalar alphaStart,
        Dscalar deltaTMax, Dscalar deltaTInc, Dscalar deltaTDec, Dscalar alphaDec, int nMin,
        Dscalar forceCutoff)
    {
    emin->setDeltaT(deltaT);
    emin->setAlphaStart(alphaStart);
    emin->setDeltaTMax(deltaTMax);
    emin->setDeltaTInc(deltaTInc);
    emin->setDeltaTDec(deltaTDec);
    emin->setAlphaDec(alphaDec);
    emin->setNMin(nMin);
    emin->setForceCutoff(forceCutoff);
    };


int main(int argc, char*argv[])
{
    int numpts = 200;
    int USE_GPU = 0;
    int USE_TENSION = 0;
    int c;
    int tSteps = 5;
    int initSteps = 0;

    Dscalar dt = 0.1;
    Dscalar p0 = 4.0;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;
    Dscalar gamma = 0.0;

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

    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    
    ForcePtr spv = make_shared<SPV2D>(numpts,1.0,4.0,reproducible);
    shared_ptr<SPV2D> SPV = dynamic_pointer_cast<SPV2D>(spv);

    EOMPtr fireMinimizer = make_shared<EnergyMinimizerFIRE>(spv);
    shared_ptr<EnergyMinimizerFIRE> FIREMIN = dynamic_pointer_cast<EnergyMinimizerFIRE>(fireMinimizer);

    spv->setCellPreferencesUniform(1.0,p0);
    spv->setv0Dr(v0,1.0);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->setEquationOfMotion(spp,spv);
    sim->setIntegrationTimestep(dt);
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    if(!initializeGPU)
        sim->setCPUOperation(true);
    sim->setReproducible(true);
    //initialize parameters

    char dataname[256];
    sprintf(dataname,"../test.nc");
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
    ncdat.WriteState(SPV);
/*
    sim->setEquationOfMotion(fireMinimizer,spv);
    setFIREParameters(FIREMIN,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
    for (int ii = 0; ii < initSteps; ++ii)
        {
        FIREMIN->setMaximumIterations((1000)*(1+ii));
        sim->performTimestep();
        };
*/
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv->reportMeanCellForce(false);

    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%100 ==0)
            {
            printf("timestep %i\n",ii);
            ncdat.WriteState(SPV);
            };
        sim->performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
  cout << spv->reportq() << endl;


    ncdat.ReadState(SPV,0,true);
    ncdat.WriteState(SPV);
    if(initializeGPU)
        cudaDeviceReset();

    SPV->computeGeometryCPU();
    Dscalar2 ans;
    ans = SPV->dAidrj(26,25);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dAidrj(26,26);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dAidrj(26,32);
    printf("%g\t%g\n",ans.x,ans.y);

    ans = SPV->dPidrj(26,25);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dPidrj(26,26);
    printf("%g\t%g\n",ans.x,ans.y);
    ans = SPV->dPidrj(26,32);
    printf("%g\t%g\n",ans.x,ans.y);

    neighborType nt = neighborType::self;
    neighborType nt1 = neighborType::first;
    neighborType nt2 = neighborType::second;
    Matrix2x2 test = SPV->d2Edridrj(26,26,nt,1.0,1.0);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    
    test = SPV->d2Edridrj(26,25,nt1,1.0,1.0);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);

    test = SPV->d2Edridrj(26,53,nt2,1.0,1.0);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    /*
    test = SPV->d2Edridrj(26,49,nt1);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    test = SPV->d2Edridrj(26,32,nt1);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    test = SPV->d2Edridrj(26,78,nt1);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    test = SPV->d2Edridrj(26,20,nt1);
    printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
    */
    //test = SPV->d2Edridrj(78,20,nt1);
  //  printf("\n\n%g\t%g\n%g\t%g\n\n",test.x11,test.x12,test.x21,test.x22);
/*
    SPV->computeForces();
    GPUArray<Dscalar2> fs;
    fs.resize(numpts);
    SPV->getForces(fs);
    ArrayHandle<Dscalar2> hf(fs);
    printf("Force on cell %i: (%f,%f)\n",26,hf.data[26].x,hf.data[26].y);
*/

   /*
    EigMat D(4);
    D.placeElementSymmetric(0,0,1.);
    D.placeElementSymmetric(1,1,2.);
    D.placeElementSymmetric(2,2,4.);
    D.placeElementSymmetric(3,3,5.);
    D.placeElementSymmetric(0,2,3.);
    D.SASolve();
    for (int ee = 0; ee < 4; ++ee)
        printf("%f\t",D.eigenvalues[ee]);
    cout <<endl;
*/

/*
    vector<int2> rowCols;
    vector<Dscalar> entries;
    SPV->getDynMatEntries(rowCols,entries,1.0,0.0);
    printf("Number of partial entries: %i\n",rowCols.size());
    EigMat D(2*numpts);
    for (int ii = 0; ii < rowCols.size(); ++ii)
        {
        int2 ij = rowCols[ii];
        D.placeElementSymmetric(ij.x,ij.y,entries[ii]);
        };

    D.SASolve();
    for (int ee = 0; ee < 4; ++ee)
        printf("%f\t",D.eigenvalues[ee]);
    cout <<endl;

    for (int jj = 0; jj < 2*numpts; ++jj)
        {
        Dscalar sum = 0.0;
        for (int ii = 0; ii < 2*numpts; ++ii)
            sum += D.mat(jj,ii);
        printf("%.2f\t",sum);
        };
    printf("\n");
*/

    return 0;
};
