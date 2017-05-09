#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "spv2d.h"
#include "selfPropelledParticleDynamics.h"
#include "EnergyMinimizerFIRE2D.h"
#include "EnergyMinimizerNewtonRaphson.h"
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
    Dscalar KA = 1.0;
    Dscalar thresh = 1e-12;

    int program_switch = 0;
    while((c=getopt(argc,argv,"k:n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'x': USE_TENSION = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'k': KA = atof(optarg); break;
            case 's': gamma = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
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

    std::cout << std::fixed;
    std::cout << std::setprecision(9);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield ); 
    clock_t t1,t2;
    bool reproducible = false;
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
    spv->setModuliUniform(KA,1.0);
    spv->setv0Dr(v0,1.0);
    printf("initializing with KA = %f\t p_0 = %f\n",KA,p0);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
//    sim->setEquationOfMotion(spp,spv);
//    sim->setIntegrationTimestep(dt);
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags

    char dataname[256];
    sprintf(dataname,"../DOS_N%i_p%.3f_KA%.1f.txt",numpts,p0,KA);
    
//    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);
//    ncdat.WriteState(SPV);

    sim->setEquationOfMotion(fireMinimizer,spv);
    if(!initializeGPU)
        sim->setCPUOperation(true);
    sim->setReproducible(true);
    //initialize parameters
    Dscalar astart, adec, tdec, tinc; int nmin;
    nmin = 5;
    astart = 0.1;
    adec= 0.99;
    tinc = 1.1;
    tdec = 0.5;
    setFIREParameters(FIREMIN,dt,0.99,50*dt,1.1,0.95,.95,20,thresh);
    setFIREParameters(FIREMIN,dt,astart,50*dt,tinc,tdec,adec,nmin,thresh);
    t1=clock();

if(program_switch ==5)
    {
    Dscalar mf;
    for (int ii = 0; ii < initSteps; ++ii)
        {
        FIREMIN->setMaximumIterations((tSteps)*(1+ii));
        sim->performTimestep();
        SPV->computeGeometryCPU();
        SPV->computeForces();
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
    return 0;
    };

if(program_switch ==4)
    {
        Dscalar mf;
        for (int ii = 0; ii < initSteps; ++ii)
            {
            FIREMIN->setMaximumIterations((tSteps)*(1+ii));
            sim->performTimestep();
            SPV->computeGeometryCPU();
            SPV->computeForces();
            mf = spv->getMaxForce();
            printf("maxForce = %g\n",mf);
            if (mf < thresh)
                break;
            };
    Dscalar mf1 = spv->getMaxForce();
    DEBUGCODEHELPER;
    printf("pre-NR max force:%e \n",mf1);
    GPUArray<Dscalar2> f;
    SPV->getForces(f);
    Eigen::VectorXd eGradient,eDisp;
    eGradient.resize(2*numpts);
    eDisp.resize(2*numpts);
    ArrayHandle<Dscalar2> hf(f);
    for (int nn = 0; nn < numpts; ++nn)
        {
        eGradient[2*nn] = hf.data[nn].x;
        eGradient[2*nn+1] = hf.data[nn].x;
        };
    vector<int2> rCs;
    vector<Dscalar> es;
    SPV->getDynMatEntries(rCs,es,1.0,1.0);
    EigMat Ds(2*numpts);
    for (int ii = 0; ii < rCs.size(); ++ii)
        {
        int2 ij = rCs[ii];
        if (ij.x == ij.y) es[ii] += 0.*1e-16;
        Ds.placeElementSymmetric(ij.x,ij.y,-es[ii]);
        };
    Ds.SASolve();
    Eigen::MatrixXd Hinv = Ds.es.eigenvectors().inverse();
    eDisp = (Hinv)*eGradient;

    GPUArray<Dscalar2> displacement;
    displacement.resize(numpts);
    {
    ArrayHandle<Dscalar2> hd(displacement);
    for (int nn = 0; nn < numpts; ++nn)
        {
        hd.data[nn].x = eDisp[2*nn];
        hd.data[nn].y = eDisp[2*nn+1];
        };
    };
    spv->moveDegreesOfFreedom(displacement);
    spv->enforceTopology();
    spv->computeForces();
    Dscalar mf2 = spv->getMaxForce();
    printf("post-NR max force:%e \n",mf2);


    Dscalar mp = spv->reportMeanP();
    Dscalar2 variances = spv->reportVarAP();
    printf("var(A) = %f\t Var(p) = %g\n",variances.x,variances.y);
    return 0;
    };


if(program_switch ==3)
    {
    printf("asd\n");
    int iter = 0;
    for (Dscalar pp = 3.8; pp < 3.88; pp+=0.005)
        {
        SPV->setCellPreferencesUniform(1.0,pp);
            SPV->computeGeometryCPU();
            SPV->computeForces();
        SPV->setModuliUniform(KA,1.0);
        printf("%f asd\n",pp);
        sim->setCurrentTimestep(0);
        Dscalar mf;
        for (int ii = 0; ii < initSteps; ++ii)
            {
    setFIREParameters(FIREMIN,dt,0.99,100*dt,1.1,0.95,.95,4,thresh);
            FIREMIN->setMaximumIterations((iter+1)*(tSteps)*(1+ii));
            sim->performTimestep();
            SPV->computeGeometryCPU();
            SPV->computeForces();
            mf = spv->getMaxForce();
            printf("maxForce = %g\n",mf);
            if (mf < thresh)
                break;
            };
        iter +=1;
        };


    Dscalar mp = spv->reportMeanP();
    Dscalar2 variances = spv->reportVarAP();
    printf("var(A) = %f\t Var(p) = %g\n",variances.x,variances.y);
    return 0;
    };

    Dscalar mf;
    for (int ii = 0; ii < initSteps; ++ii)
        {
        if (ii > 0 && mf >9e-6) return 0;
        FIREMIN->setMaximumIterations((tSteps)*(1+ii));
        sim->performTimestep();
        SPV->computeGeometryCPU();
        SPV->computeForces();
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

if(program_switch ==2)
    {
    Dscalar mp = spv->reportMeanP();
    Dscalar2 variances = spv->reportVarAP();
    printf("var(A) = %f\t Var(p) = %g\n",variances.x,variances.y);
    sprintf(dataname,"../shapeData_N%i.txt",numpts);
    ofstream outfile;
    outfile.open(dataname,std::ios_base::app);
    outfile <<fixed << setprecision(9) << p0 <<"\t" << KA <<"\t" << mp << "\t";
    outfile << variances.x << "\t" <<variances.y << "\t";
    outfile << mf << "\n" ;
    return 0;
    };

if (program_switch ==1)
    {
    sprintf(dataname,"../qData_N%i.txt",numpts);
    ofstream outfile;
    outfile.open(dataname,std::ios_base::app);
    outfile << p0 <<"\t" << meanQ << "\t" << varQ  <<"\n";
    return 0;
    };

    if (mf > thresh) return 0;

//    ncdat.ReadState(SPV,0,true);
//    ncdat.WriteState(SPV);
    if(initializeGPU)
        cudaDeviceReset();

    SPV->computeGeometryCPU();
    Dscalar2 ans;

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
    vector<int2> rowCols;
    vector<Dscalar> entries;
    SPV->getDynMatEntries(rowCols,entries,1.0,1.0);
    printf("Number of partial entries: %lu\n",rowCols.size());
    EigMat D(2*numpts);
    for (int ii = 0; ii < rowCols.size(); ++ii)
        {
        int2 ij = rowCols[ii];
        D.placeElementSymmetric(ij.x,ij.y,entries[ii]);
        };

    int evecTest = 11;
    D.SASolve(evecTest+1);
    for (int ee = 0; ee < 40; ++ee)
        printf("%f\t",D.eigenvalues[ee]);
    cout <<endl;

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

    Dscalar E0 = spv->quadraticEnergy();
    printf("initial energy = %g\n",spv->quadraticEnergy());
    spv->computeForces();
    printf("initial energy = %g\n",spv->quadraticEnergy());
    spv->moveDegreesOfFreedom(disp);
    spv->computeForces();
    Dscalar E1 = spv->quadraticEnergy();
    printf("positive delta energy = %g\n",spv->quadraticEnergy());
    spv->moveDegreesOfFreedom(dispneg);
    spv->moveDegreesOfFreedom(dispneg);
    spv->computeForces();
    Dscalar E2 = spv->quadraticEnergy();
    printf("negative delta energy = %g\n",spv->quadraticEnergy());
    printf("differences: %f\t %f\n",E1-E0,E2-E0);
    printf("der = %f\n",(E1+E2-2.0*E0)/(mag*mag));


if (program_switch ==0)
    {
    ofstream outfile;
    outfile.open(dataname,std::ios_base::app);
        for (int ee = 0; ee < 2*numpts-1; ++ee)
            {
            Dscalar temp = D.eigenvalues[ee];
            if (temp > 0)
                temp = sqrt(temp);
            else
                temp = 0.;
            outfile << temp <<"\n";
            };

    //unstressed version?
    vector<int2> rowColsU;
    vector<Dscalar> entriesU;
    SPV->getDynMatEntries(rowColsU,entriesU,1.0,0.0);
    EigMat DU(2*numpts);
    for (int ii = 0; ii < rowColsU.size(); ++ii)
        {
        int2 ij = rowColsU[ii];
        DU.placeElementSymmetric(ij.x,ij.y,entriesU[ii]);
        };

    DU.SASolve();
    for (int ee = 0; ee < 40; ++ee)
        printf("%f\t",DU.eigenvalues[ee]);
    cout <<endl;
    
    char datanameU[256];
    sprintf(datanameU,"../DOS_unstressed_N%i_p%.3f_KA%.1f.txt",numpts,p0,KA);
    ofstream outfileU;
    outfileU.open(datanameU,std::ios_base::app);
    for (int ee = 0; ee < 2*numpts-1; ++ee)
        {
        Dscalar temp = DU.eigenvalues[ee];
        if (temp > 0)
            temp = sqrt(temp);
        else
            temp = 0.;
        outfileU << temp <<"\n";
        };
    };
    return 0;
};
