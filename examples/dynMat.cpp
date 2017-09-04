#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "EnergyMinimizerFIRE2D.h"
#include "DatabaseNetCDFSPV.h"
#include "eigenMatrixInterface.h"

/*!
This file compiles to produce an executable that demonstrates a simple example of using the Eigen
interface to diagonalize a dynamical matrix.
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
    //as in the examples in the main directory, there are a bunch of default parameters that
    //can be changed from the command line
    int numpts = 200;
    int USE_GPU = 0;
    int c;
    int tSteps = 5;
    int initSteps = 0;

    Dscalar dt = 0.1;
    Dscalar p0 = 4.0;
    Dscalar pf = 4.0;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;
    Dscalar KA = 1.0;
    Dscalar thresh = 1e-12;

    //This example is a bit more ragged than the others, and program_switch has been abused for testing features that have not been cleaned up yet
    int program_switch = 0;
    while((c=getopt(argc,argv,"k:n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:q:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'k': KA = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'q': pf = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'r': thresh = atof(optarg); break;
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

    //the voronoi model set up is just as before
    shared_ptr<VoronoiQuadraticEnergy> spv = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);
    //..and instead of a self-propelled cell equation of motion, we use a FIRE minimizer
    shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(spv);

    spv->setCellPreferencesUniform(1.0,p0);
    spv->setModuliUniform(KA,1.0);
    printf("initializing with KA = %f\t p_0 = %f\n",KA,p0);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(fireMinimizer,spv);
    sim->setCPUOperation(!initializeGPU);

    //initialize FIRE parameters...these parameters are pretty standard for many MD settings, and shouldn't need too much adjustment
    Dscalar astart, adec, tdec, tinc; int nmin;
    nmin = 5;
    astart = .1;
    adec= 0.99;
    tinc = 1.1;
    tdec = 0.5;
    setFIREParameters(fireMinimizer,dt,astart,50*dt,tinc,tdec,adec,nmin,thresh);
    t1=clock();

    //test minimization simply
    if(program_switch ==5)
        {
        Dscalar mf;
        for (int ii = 0; ii < initSteps; ++ii)
            {
            fireMinimizer->setMaximumIterations((tSteps)*(1+ii));
            sim->performTimestep();
            spv->computeGeometryCPU();
            spv->computeForces();
            mf = spv->getMaxForce();
            printf("maxForce = %g\t energy/cell = %g\n",mf,spv->computeEnergy()/(Dscalar)numpts);
            if (mf < thresh)
                break;
            };

        t2=clock();
        Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC;
        cout << "minimization was ~ " << steptime << endl;
        Dscalar meanQ, varQ;
        meanQ = spv->reportq();
        varQ = spv->reportVarq();
        printf("Cell <q> = %f\t Var(q) = %g\n",meanQ,varQ);
        return 0;
        };

    //minimize to tolerance
    Dscalar mf;
    for (int ii = 0; ii < initSteps; ++ii)
        {
        if (ii > 0 && mf > 0.0001) return 0;
        fireMinimizer->setMaximumIterations((tSteps)*(1+ii));
        sim->performTimestep();
        spv->computeGeometryCPU();
        spv->computeForces();
        mf = spv->getMaxForce();
        printf("maxForce = %g\n",mf);
        if (mf < thresh)
            break;
        };

    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC;
    cout << "minimization was ~ " << steptime << endl;
    Dscalar meanQ, varQ;
    meanQ = spv->reportq();
    varQ = spv->reportVarq();
    printf("Cell <q> = %f\t Var(q) = %g\n",meanQ,varQ);

    printf("Finished with initialization\n");
    //cout << "current q = " << spv.reportq() << endl;
    spv->reportMeanCellForce(false);
    if (mf > thresh) return 0;

    //build the dynamical matrix
    spv->computeGeometryCPU();
    vector<int2> rowCols;
    vector<Dscalar> entries;
    spv->getDynMatEntries(rowCols,entries,1.0,1.0);
    printf("Number of partial entries: %lu\n",rowCols.size());
    EigMat D(2*numpts);
    for (int ii = 0; ii < rowCols.size(); ++ii)
        {
        int2 ij = rowCols[ii];
        D.placeElementSymmetric(ij.x,ij.y,entries[ii]);
        };

    int evecTest = 11;
    D.SASolve(evecTest+1);
    vector<Dscalar> eigenv;
    for (int ee = 0; ee < 40; ++ee)
        {
        D.getEvec(ee,eigenv);
        printf("lambda = %f\t \n",D.eigenvalues[ee]);
        };
    cout <<endl;

    //compare with a numerical derivative
    GPUArray<Dscalar2> disp,dispneg;
    disp.resize(numpts);
    dispneg.resize(numpts);
    Dscalar mag = 1e-2;
    {
    ArrayHandle<Dscalar2> hp(disp);
    ArrayHandle<Dscalar2> hn(dispneg);
    for (int ii = 0; ii < numpts; ++ii)
        {
        hp.data[ii].x = mag*D.eigenvectors[evecTest][2*ii];
        hp.data[ii].y = mag*D.eigenvectors[evecTest][2*ii+1];
        hn.data[ii].x = -mag*D.eigenvectors[evecTest][2*ii];
        hn.data[ii].y = -mag*D.eigenvectors[evecTest][2*ii+1];
        //printf("(%f,%f)\t",hn.data[ii].x,hn.data[ii].y);
        };
    };

    Dscalar E0 = spv->computeEnergy();
    printf("initial energy = %g\n",spv->computeEnergy());
    spv->computeForces();
    printf("initial energy = %g\n",spv->computeEnergy());
    spv->moveDegreesOfFreedom(disp);
    spv->computeForces();
    Dscalar E1 = spv->computeEnergy();
    printf("positive delta energy = %g\n",spv->computeEnergy());
    spv->moveDegreesOfFreedom(dispneg);
    spv->moveDegreesOfFreedom(dispneg);
    spv->computeForces();
    Dscalar E2 = spv->computeEnergy();
    printf("negative delta energy = %g\n",spv->computeEnergy());
    printf("differences: %f\t %f\n",E1-E0,E2-E0);
    printf("der = %f\n",(E1+E2-2.0*E0)/(mag*mag));

    return 0;
};
