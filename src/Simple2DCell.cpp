#define ENABLE_CUDA

#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
/*! \file Simple2DCell.cpp */

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
Set the Area and Perimeter preferences to the input vector
*/
void Simple2DCell::setCellPreferences(vector<Dscalar2> &APPref)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = APPref[ii].x;
        h_p.data[ii].y = APPref[ii].y;
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
    random_device rd;
    mt19937 gen(rand());
    mt19937 genrd(rd());
    uniform_real_distribution<Dscalar> uniRand(0.0,boxsize);

    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Dscalar x, y;
        if(Reproducible)
            {
            x = uniRand(gen);
            y = uniRand(gen);
            }
        else
            {
            x = uniRand(genrd);
            y = uniRand(genrd);
            };
        h_p.data[ii].x = x;
        h_p.data[ii].y = y;
        };
    };

/*!
Does not update any other lists -- it is the user's responsibility to maintain topology, etc, when
using this function.
*/
void Simple2DCell::setCellPositions(vector<Dscalar2> newCellPositions)
    {
    Ncells = newCellPositions.size();
    if(cellPositions.getNumElements() != Ncells) cellPositions.resize(Ncells);
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        h_p.data[ii] = newCellPositions[ii];
    }

/*!
Does not update any other lists -- it is the user's responsibility to maintain topology, etc, when
using this function.
*/
void Simple2DCell::setVertexPositions(vector<Dscalar2> newVertexPositions)
    {
    Nvertices = newVertexPositions.size();
    if(vertexPositions.getNumElements() != Nvertices) vertexPositions.resize(Nvertices);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        h_v.data[ii] = newVertexPositions[ii];
    }


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
 * set all cell types to i
 */
void Simple2DCell::setCellTypeUniform(int i)
    {
    CellType.resize(Ncells);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ct.data[ii] = i;
        };
    };


/*!
 \param types a vector of integers that the cell types will be set to
 */
void Simple2DCell::setCellType(vector<int> &types)
    {
    CellType.resize(Ncells);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ct.data[ii] = types[ii];
        };
    };

/*!
This function allows a user to set the vertex topology by hand. The user is responsible for making
sure the input topology is sensible. DMS NOTE -- this functionality has not been thoroughly tested
\pre Nvertices and vertex positions are already set
\param cellVertexIndices a vector of vector of ints. Each vector of ints must correspond to the
counter-clockwise ordering of vertices that make up the cell, and every vertex should appear at most
three times in different cells
*/
void Simple2DCell::setVertexTopologyFromCells(vector< vector<int> > cellVertexIndices)
    {
    //set the number of cells, number of vertices per cell, and the maximum number of vertices per
    //cell from the input
    Ncells = cellVertexIndices.size();
    cellVertexNum.resize(Ncells);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::overwrite);
    vertexMax = 0;
    for (int cc = 0; cc < Ncells; ++cc)
        {
        if(cellVertexIndices[cc].size() > vertexMax)
            vertexMax = cellVertexIndices[cc].size();
        h_cvn.data[cc] = cellVertexIndices[cc].size();
        };
    vertexMax +=2;
    //set the vertices associated with every cell
    n_idx = Index2D(vertexMax,Ncells);
    cellVertices.resize(vertexMax*Ncells);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::overwrite);
    for (int cc = 0; cc < Ncells; ++cc)
        {
        for (int nn = 0; nn < cellVertexIndices[cc].size(); ++nn)
            {
            h_cv.data[n_idx(nn,cc)] = cellVertexIndices[cc][nn];
            };
        };

    //deduce the vertex-vertex  and vertex-cell connections from the input
    vector<int> vertexNeighborsFound(Nvertices,0);
    vector<int> vertexCellNeighborsFound(Nvertices,0);
    vertexNeighbors.resize(3*Nvertices);
    vertexCellNeighbors.resize(3*Nvertices);
    ArrayHandle<int> h_vn(vertexNeighbors,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::overwrite);
    int vlast, vcur, vnext;
    for (int cc = 0; cc < Ncells; ++cc)
        {
        int neighs = cellVertexIndices[cc].size();
        //iterate through the cell list, starting with vnext being the first item in the list
        vlast = cellVertexIndices[cc][neighs-2];
        vcur = cellVertexIndices[cc][neighs-1];
        for (int nn = 0; nn < neighs; ++nn)
            {
            vnext = cellVertexIndices[cc][nn];
            //find vertex-vertex neighbors
            if(vertexNeighborsFound[vcur] < 3)
                {
                bool addVLast = true;
                bool addVNext = true;
                for (int vv = 0; vv < vertexNeighborsFound[vcur]; ++vv)
                    {
                    if(h_vn.data[3*vcur+vv] == vlast) addVLast = false;
                    if(h_vn.data[3*vcur+vv] == vnext) addVNext = false;
                    };
                if(addVLast)
                    {
                    h_vn.data[3*vcur + vertexNeighborsFound[vcur]] = vlast;
                    vertexNeighborsFound[vcur] += 1;
                    };
                if(addVNext)
                    {
                    h_vn.data[3*vcur + vertexNeighborsFound[vcur]] = vnext;
                    vertexNeighborsFound[vcur] += 1;
                    };
                };
            //find vertex-cell neighbors
            h_vcn.data[3*vcur + vertexCellNeighborsFound[vcur]] = cc;
            vertexCellNeighborsFound[vcur] += 1;
            // advance the loop
            vlast = vcur;
            vcur = vnext;
            };
        };
    };


/*!
 *Sets the size of itt, tti, idxToTag, and tagToIdx, and sets all of them so that
 array[i] = i,
 i.e., unsorted
 \pre Ncells is determined
 */
void Simple2DCell::initializeCellSorting()
    {
    itt.resize(Ncells);
    tti.resize(Ncells);
    idxToTag.resize(Ncells);
    tagToIdx.resize(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        itt[ii]=ii;
        tti[ii]=ii;
        idxToTag[ii]=ii;
        tagToIdx[ii]=ii;
        };
    };

/*!
 *Sets the size of ittVertex, ttiVertex, idxToTagVertex, and tagToIdxVertex,and sets all of them so that
 array[i] = i,
 i.e., things are unsorted
 \pre Nvertices is determined
 */
void Simple2DCell::initializeVertexSorting()
    {
    ittVertex.resize(Nvertices);
    ttiVertex.resize(Nvertices);
    idxToTagVertex.resize(Nvertices);
    tagToIdxVertex.resize(Nvertices);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        ittVertex[ii]=ii;
        ttiVertex[ii]=ii;
        idxToTagVertex[ii]=ii;
        tagToIdxVertex[ii]=ii;
        };

    };

/*!
 * Always called after spatial sorting is performed, reIndexCellArray shuffles the order of an array
    based on the spatial sort order of the cells
*/
void Simple2DCell::reIndexCellArray(GPUArray<Dscalar2> &array)
    {
    GPUArray<Dscalar2> TEMP = array;
    ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
Re-indexes GPUarrays of Dscalars
*/
void Simple2DCell::reIndexCellArray(GPUArray<Dscalar> &array)
    {
    GPUArray<Dscalar> TEMP = array;
    ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
Re-indexes GPUarrays of ints
*/
void Simple2DCell::reIndexCellArray(GPUArray<int> &array)
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
 * Called if the vertices need to be spatially sorted, reIndexVertexArray shuffles the order of an
 * array based on the spatial sort order of the vertices
*/
void Simple2DCell::reIndexVertexArray(GPUArray<Dscalar2> &array)
    {
    GPUArray<Dscalar2> TEMP = array;
    ArrayHandle<Dscalar2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        ar.data[ii] = temp.data[ittVertex[ii]];
        };
    };

void Simple2DCell::reIndexVertexArray(GPUArray<Dscalar> &array)
    {
    GPUArray<Dscalar> TEMP = array;
    ArrayHandle<Dscalar> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        ar.data[ii] = temp.data[ittVertex[ii]];
        };
    };

void Simple2DCell::reIndexVertexArray(GPUArray<int> &array)
    {
    GPUArray<int> TEMP = array;
    ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        ar.data[ii] = temp.data[ittVertex[ii]];
        };
    };

/*!
 * take the current location of the cells and sort them according the their order along a 2D Hilbert curve
 */
void Simple2DCell::spatiallySortCells()
    {
    //itt and tti are the changes that happen in the current sort
    //idxToTag and tagToIdx relate the current indexes to the original ones
    HilbertSorter hs(Box);

    vector<pair<int,int> > idxCellSorter(Ncells);

    //sort points by Hilbert Curve location
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxCellSorter[ii].first=hs.getIdx(h_p.data[ii]);
        idxCellSorter[ii].second = ii;
        };
    sort(idxCellSorter.begin(),idxCellSorter.end());

    //update tti and itt
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int newidx = idxCellSorter[ii].second;
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
    reIndexCellArray(cellPositions);
    reIndexCellArray(Moduli);
    reIndexCellArray(AreaPeriPreferences);
    reIndexCellArray(CellType);
    };

/*!
 * take the current location of the vertices and sort them according the their order along a 2D
 * Hilbert curve. This routine first sorts the vertices, and then uses the vertex sorting to derive
 * a sorting of the cells
 * \post both the itt, tti,... and ittVertex, ttiVertex... arrays are correctly set
 */
void Simple2DCell::spatiallySortVertices()
    {
    //ittVertex and ttiVertex are the changes that happen in the current sort
    //idxToTagVertex and tagToIdxVertex relate the current indexes to the original ones
    HilbertSorter hs(Box);

    vector<pair<int,int> > idxSorterVertex(Nvertices);

    //sort points by Hilbert Curve location
    ArrayHandle<Dscalar2> h_p(vertexPositions,access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        idxSorterVertex[ii].first=hs.getIdx(h_p.data[ii]);
        idxSorterVertex[ii].second = ii;
        };
    sort(idxSorterVertex.begin(),idxSorterVertex.end());

    //update tti and itt
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        int newidx = idxSorterVertex[ii].second;
        ittVertex[ii] = newidx;
        ttiVertex[newidx] = ii;
        };

    //update points, idxToTag, and tagToIdx
    vector<int> tempi = idxToTagVertex;
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        idxToTagVertex[ii] = tempi[ittVertex[ii]];
        tagToIdxVertex[tempi[ittVertex[ii]]] = ii;
        };
    reIndexVertexArray(vertexPositions);

    //grab array handles and old copies of GPUarrays
    GPUArray<int> TEMP_vertexNeighbors = vertexNeighbors;
    GPUArray<int> TEMP_vertexCellNeighbors = vertexCellNeighbors;
    GPUArray<int> TEMP_cellVertices = cellVertices;
    ArrayHandle<int> temp_vn(TEMP_vertexNeighbors,access_location::host, access_mode::read);
    ArrayHandle<int> temp_vcn(TEMP_vertexCellNeighbors,access_location::host, access_mode::read);
    ArrayHandle<int> temp_cv(TEMP_cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> vn(vertexNeighbors,access_location::host, access_mode::readwrite);
    ArrayHandle<int> vcn(vertexCellNeighbors,access_location::host, access_mode::readwrite);
    ArrayHandle<int> cv(cellVertices,access_location::host, access_mode::readwrite);
    ArrayHandle<int> cvn(cellVertexNum,access_location::host,access_mode::read);

    //Great, now use the vertex ordering to derive a cell spatial ordering
    vector<pair<int,int> > idxCellSorter(Ncells);

    vector<bool> cellOrdered(Ncells,false);
    int cellOrdering = 0;
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        if(cellOrdering == Ncells) continue;
        int vertexIndex = ittVertex[vv];
        for (int ii = 0; ii < 3; ++ii)
            {
            int cellIndex = vcn.data[3*vertexIndex +ii];
            if(!cellOrdered[cellIndex])
                {
                cellOrdered[cellIndex] = true;
                idxCellSorter[cellIndex].first=cellOrdering;
                idxCellSorter[cellIndex].second = cellIndex;
                cellOrdering += 1;
                };
            };
        };
    sort(idxCellSorter.begin(),idxCellSorter.end());

    //update tti and itt
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int newidx = idxCellSorter[ii].second;
        itt[ii] = newidx;
        tti[newidx] = ii;
        };

    //update points, idxToTag, and tagToIdx
    vector<int> tempiCell = idxToTag;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxToTag[ii] = tempiCell[itt[ii]];
        tagToIdx[tempiCell[itt[ii]]] = ii;
        };

    reIndexCellArray(cellPositions);

    //Finally, now that both cell and vertex re-indexing is known, update auxiliary data structures
    //Start with everything that can be done with just the cell indexing
    reIndexCellArray(Moduli);
    reIndexCellArray(AreaPeriPreferences);
    reIndexCellArray(CellType);
    reIndexCellArray(cellVertexNum);
    //Now the rest
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        int vertexIndex = ttiVertex[vv];
        for (int ii = 0; ii < 3; ++ii)
            {
            vn.data[3*vertexIndex+ii] = ttiVertex[temp_vn.data[3*vv+ii]];
            vcn.data[3*vertexIndex+ii] = tti[temp_vcn.data[3*vv+ii]];
            };
        };

    for (int cc = 0; cc < Ncells; ++cc)
        {
        int cellIndex = tti[cc];
        //the cellVertexNeigh array is already sorted
        int neighs = cvn.data[cellIndex];
        for (int nn = 0; nn < neighs; ++nn)
            cv.data[n_idx(nn,cellIndex)] = ttiVertex[temp_cv.data[n_idx(nn,cc)]];
        };
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

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A-A_0)^2 + K_P(P-P_0)^2
*/
Dscalar Simple2DCell::quadraticEnergy()
    {
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    Dscalar energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        energy += KA * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        energy += KP * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };

    Energy = energy;
    return energy;
    };
