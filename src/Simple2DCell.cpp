#define ENABLE_CUDA

#include "Simple2DCell.h"
#include "Simple2DCell.cuh"

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DCell::Simple2DCell() :
    Ncells(0), Nvertices(0),GPUcompute(true)
    {
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

/*!
Resize the box so that every cell has, on average, area = 1, and place cells via a simple,
reproducible RNG
*/
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
set all cell K_A, K_P preferences to uniform values.
PLEASE NOTE that as an optimization this data is not actually used at the moment,
but the code could be trivially altered to use this
*/
void Simple2DCell::setModuliUniform(Dscalar newKA, Dscalar newKP)
    {
    KA=newKA;
    KP=newKP;
    Moduli.resize(Ncells);
    ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_m.data[ii].x = KA;
        h_m.data[ii].y = KP;
        };
    };


/*!
 * Always called after spatial sorting is performed, reIndexArrays shuffles the order of an array
    based on the spatial sort order of the cells
*/
void Simple2DCell::reIndexArray(GPUArray<Dscalar2> &array)
    {
    GPUArray<Dscalar2> TEMP = array;
    ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

void Simple2DCell::reIndexArray(GPUArray<Dscalar> &array)
    {
    GPUArray<Dscalar> TEMP = array;
    ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

void Simple2DCell::reIndexArray(GPUArray<int> &array)
    {
    GPUArray<int> TEMP = array;
    ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
 * take the current location of the points and sort them according the their order along a 2D Hilbert curve
 */
void Simple2DCell::spatiallySortPoints()
    {
    //itt and tti are the changes that happen in the current sort
    //idxToTag and tagToIdx relate the current indexes to the original ones
    HilbertSorter hs(Box);

    vector<pair<int,int> > idxSorter(Ncells);

    //sort points by Hilbert Curve location
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxSorter[ii].first=hs.getIdx(h_p.data[ii]);
        idxSorter[ii].second = ii;
        };
    sort(idxSorter.begin(),idxSorter.end());

    //update tti and itt
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int newidx = idxSorter[ii].second;
        itt[ii] = newidx;
        tti[newidx] = ii;
        };

    //update points, idxToTag, and tagToIdx
    vector<int> tempi = idxToTag;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxToTag[ii] = tempi[itt[ii]];
        tagToIdx[tempi[itt[ii]]] = ii;
        };
    reIndexArray(cellPositions);
    };


/*!
a utility/testing function...output the currently computed mean net force to screen.
\param verbose if true also print out the force on each cell
*/
void Simple2DCell::reportMeanCellForce(bool verbose)
    {
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> p(cellPositions,access_location::host,access_mode::read);
    Dscalar fx = 0.0;
    Dscalar fy = 0.0;
    Dscalar min = 10000;
    Dscalar max = -10000;
    for (int i = 0; i < Ncells; ++i)
        {
        if (h_f.data[i].y >max)
            max = h_f.data[i].y;
        if (h_f.data[i].x >max)
            max = h_f.data[i].x;
        if (h_f.data[i].y < min)
            min = h_f.data[i].y;
        if (h_f.data[i].x < min)
            min = h_f.data[i].x;
        fx += h_f.data[i].x;
        fy += h_f.data[i].y;

        if(verbose)
            printf("cell %i: \t position (%f,%f)\t force (%e, %e)\n",i,p.data[i].x,p.data[i].y ,h_f.data[i].x,h_f.data[i].y);
        };
    if(verbose)
        printf("min/max force : (%f,%f)\n",min,max);
    printf("Mean force = (%e,%e)\n" ,fx/Ncells,fy/Ncells);
    };


