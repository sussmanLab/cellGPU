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
    setDeltaT(deltaT);
    initialize();
    };

/*!
This function should always be called by the user using k=0 (the default value). The function will
recursively call itself with higher k-values as needed.
*/
void autocorrelator::add(Dscalar w, int k)
    {
    };

/*!
if the boolean argument is set to true, the mean value will be subtracted off from the autocorrelation...
this is a necessary part for computing some autocorrelations, e.g., the MSD
*/
void autocorrelator::evaluate(bool normalize)
    {
    };

/*!
initializes the necessary data structures
*/
void autocorrelator::initialize()
    {
    };

/*!
If a value is added so that the number of correlator levels needs to grow, this function will be
called to do this;
*/
void autocorrelator::growCorrelationLevel()
    {
    };
