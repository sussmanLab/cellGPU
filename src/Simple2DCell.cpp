#define ENABLE_CUDA

#include "Simple2DCell.h"
#include "Simple2DCell.cuh"

/*!

*/
Simple2DCell::Simple2DCell()
    {
    Ncells = 0;
    Nvertices = 0;
    GPUcompute = true;
    };
/*!
Generically believe that cells in 2D have a notion of a preferred area and perimeter
*/
void Simple2DCell::setCellPreferencesUniform(Dscalar A0, Dscalar P0)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = A0;
        h_p.data[ii].y = P0;
        };
    };

void Simple2DCell::setCellPositionsRandomly()
    {
    Dscalar boxsize = sqrt((Dscalar)Ncells);
    Box.setSquare(boxsize,boxsize);
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Dscalar x =EPSILON+boxsize/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
        Dscalar y =EPSILON+boxsize/(Dscalar)(RAND_MAX)* (Dscalar)(rand()%RAND_MAX);
        if(x >=boxsize) x = boxsize-EPSILON;
        if(y >=boxsize) y = boxsize-EPSILON;
        h_p.data[ii].x = x;
        h_p.data[ii].y = y;
        };
    };

    
/*!
\param i the value of the offset that should be sent to the cuda RNG...
\param gs the global seed to use
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void Simple2DCell::initializeCurandStates(int gs, int i)
    {
    ArrayHandle<curandState> d_curandRNGs(cellRNGs,access_location::device,access_mode::overwrite);
    int globalseed = gs;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_curand(d_curandRNGs.data,Ncells,i,globalseed);
    };

