#define ENABLE_CUDA

#include "autocorrelator.h"
/*! \file autocorrelator.cpp */

/*!
The base constructor determines the number of points per correlator level, the number of points over
which to average, and the time spacing rate at which values will be added
*/
autocorrelator::autocorrelator(int pp, int mm, Dscalar deltaT)
    {
    p=pp;
    m=mm;
    minimumDistance = p/m;
    setDeltaT(deltaT);
    initialize();
    };

/*!
initializes the necessary data structures
*/
void autocorrelator::initialize()
    {
    nCorrelators = 0;
    vector<int> pZeroInt(p,0);
    vector<Dscalar> pZeroDscalar(p,0.0);

    accumulatedValue = 0.0;
    growCorrelationLevel();
    };

/*!
If a value is added so that the number of correlator levels needs to grow, this function will be
called to do this;
*/
void autocorrelator::growCorrelationLevel()
    {
    vector<int> pZeroInt(p,0);
    vector<Dscalar> pZeroDscalar(p,0.0);
    vector<Dscalar> pShiftDscalar(p,-20000000000.0);
    shift.push_back(pShiftDscalar);
    correlation.push_back(pZeroDscalar);
    nCorrelation.push_back(pZeroInt);
    nAccumulator.push_back(0);
    accumulator.push_back(0.0);
    insertIndex.push_back(0);

    nCorrelators+=1;
    };

/*!
This function should always be called by the user using k=0 (the default value). The function will
recursively call itself with higher k-values as needed.
*/
void autocorrelator::add(Dscalar w, int k)
    {
    //If we exceed the number of correlator levels, grow the lists
    if (k == nCorrelators)
        growCorrelationLevel();
    
    //insert into shift array
    shift[k][insertIndex[k]]=w;

    //add to the average value
    if(k==0)
        accumulatedValue += w;

    //add to accumulator, and, if needed, recursively call the higher-level add
    accumulator[k] +=w;
    nAccumulator[k] +=1;
    if(nAccumulator[k] == m)
        {
        add(accumulator[k]/m,k+1);
        accumulator[k]=0.0;
        nAccumulator[k]=0;
        };

    //calculate the correlations...the first level has to be handled differently
    int index1 = insertIndex[k];
    if (k==0)
        {
        int index2 = index1;
        for (int jj = 0; jj < p; ++jj)
            {
            if (shift[k][index2] > -10000000000.0)
                {
                correlation[k][jj] +=shift[k][index1]*shift[k][index2];
                nCorrelation[k][jj] +=1;
                };
            index2 -= 1;
            if (index2 < 0) index2 +=p;
            };
        }
    else
        {
        int index2 = index1-minimumDistance;
        for (int jj = minimumDistance; jj < p; ++jj)
            {
            if (index2 < 0) index2 +=p;
            if (shift[k][index2] > -10000000000.0)
                {
                correlation[k][jj] +=shift[k][index1]*shift[k][index2];
                nCorrelation[k][jj] +=1;
                };
            index2 -= 1;
            };
        };
    insertIndex[k] +=1;
    if (insertIndex[k] == p)
        insertIndex[k] = 0;
    };

/*!
if the boolean argument is set to true, the mean value will be subtracted off from the autocorrelation...
this is a necessary part for computing some autocorrelations, e.g., the MSD
*/
void autocorrelator::evaluate(bool normalize)
    {
    Dscalar auxiliary = 0.0;
    if (normalize)
        auxiliary = (accumulatedValue/nCorrelation[0][0])*(accumulatedValue/nCorrelation[0][0]);

    //the first level of the correlator is handled differently from the others
    for (int ii = 0; ii < p; ++ii)
        if(nCorrelation[0][ii] > 0)
            {
            Dscalar autocorr = correlation[0][ii]/nCorrelation[0][ii] - auxiliary;
            correlator.push_back(make_Dscalar2(ii*dt,autocorr));
            };

    //the other levels are all handled the same way
    for (int k = 1; k < nCorrelators; ++k)
        for (int ii = minimumDistance; ii < p; ++ii)
            if (nCorrelation[k][ii] > 0)
                {
                Dscalar autocorr = correlation[k][ii]/nCorrelation[k][ii] - auxiliary;
                Dscalar time = dt*ii*pow((Dscalar)m,k);
                correlator.push_back(make_Dscalar2(time,autocorr));
                };
    };
