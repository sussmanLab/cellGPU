#include "EnergyMinimizerNewtonRaphson.h"

/*! \file ENergyMinimizerNewtonRaphson.cpp
*/

/*!
Initialize the minimizer with a reference to a target system, set a bunch of default parameters.
Of note, the current default is CPU operation
*/
EnergyMinimizerNewtonRaphson::EnergyMinimizerNewtonRaphson(shared_ptr<Simple2DModel> system)
    {
    DEBUGCODEHELPER;
    set2DModel(system);
    N = system->getNumberOfDegreesOfFreedom();
    cout << N << endl;
    N = State->getNumberOfDegreesOfFreedom();
    cout << N << endl;
    DEBUGCODEHELPER;
    initializeFromModel();
    DEBUGCODEHELPER;
    tether = 1e-12;
    DEBUGCODEHELPER;
    };

/*!
Initialize the minimizer with some default parameters.
\pre requires a Simple2DModel (to set N correctly) to be already known
*/
void EnergyMinimizerNewtonRaphson::initializeFromModel()
    {
    DEBUGCODEHELPER;
    N = State->getNumberOfDegreesOfFreedom();
    cout << N << endl;
    DEBUGCODEHELPER;
    force.resize(N);
    DEBUGCODEHELPER;
    displacement.resize(N);
    DEBUGCODEHELPER;
    {
    ArrayHandle<Dscalar2> h_f(force);
    Dscalar2 zero; zero.x = 0.0; zero.y = 0.0;
    for(int i = 0; i <N; ++i)
        {
        h_f.data[i]=zero;
        };
    };
    DEBUGCODEHELPER;
    EM = EigMat(2*N);
    DEBUGCODEHELPER;
    eGradient.resize(2*N);
    DEBUGCODEHELPER;
    eDisplace.resize(2*N);
    DEBUGCODEHELPER;
    Hinverse = Eigen::MatrixXd::Zero(2*N,2*N);
    DEBUGCODEHELPER;
    };

/*!

*/
void EnergyMinimizerNewtonRaphson::minimize()
    {
    DEBUGCODEHELPER;
    getGradient();
    DEBUGCODEHELPER;
    getHessian();
    DEBUGCODEHELPER;
    {
    Hinverse = EM.es.eigenvectors().inverse();
    eDisplace = Hinverse*eGradient;
    ArrayHandle<Dscalar2> hd(displacement);
    for (int nn = 0; nn < N; ++nn)
        {
        hd.data[nn].x = eDisplace[2*nn];
        hd.data[nn].y = eDisplace[2*nn+1];
        };
    };
    DEBUGCODEHELPER;
    //move particles, then update the forces
    State->moveDegreesOfFreedom(displacement);
    DEBUGCODEHELPER;
    State->enforceTopology();
    DEBUGCODEHELPER;
    State->computeForces();
    DEBUGCODEHELPER;
    Dscalar mf = State->getMaxForce();
    DEBUGCODEHELPER;
    printf("post-NR max force:%e \n",mf);
    };

/*!
*/
void EnergyMinimizerNewtonRaphson::getGradient()
    {
    State->computeForces();
    State->getForces(force);
    ArrayHandle<Dscalar2> hf(force);
    for (int nn = 0; nn < N; ++nn)
        {
        eGradient[2*nn] = hf.data[nn].x;
        eGradient[2*nn+1] = hf.data[nn].x;
        };
    };
/*!

*/
void EnergyMinimizerNewtonRaphson::getHessian()
    {
    vector<int2> rowCols;
    vector<Dscalar> entries;
    State->getDynMatEntries(rowCols,entries,1.0,1.0);
    for (int ii = 0; ii < rowCols.size(); ++ii)
        {
        int2 ij = rowCols[ii];
        if (ij.x ==ij.y)
            entries[ii] += tether;
        EM.placeElementSymmetric(ij.x,ij.y,entries[ii]);
        };
    EM.SASolve();
    };

