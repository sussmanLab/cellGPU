#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
/*! \file Simple2DCell.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DCell::Simple2DCell() :
    Ncells(0), Nvertices(0),GPUcompute(true),Energy(-1.0)
    {
    forcesUpToDate = false;
    Box = make_shared<periodicBoundaries>();
    };

/*!
Initialize the data structures to the size specified by n, and set default values.
*/
void Simple2DCell::initializeSimple2DCell(int n, bool gpu)
    {
    Ncells = n;
    Nvertices = 2*Ncells;

    if(!gpu)
        {
        cout << " disabling gpu memory use" << endl;
        cellPositions.neverGPU = true;
        vertexPositions.neverGPU = true;
        cellVelocities.neverGPU = true;
        cellMasses.neverGPU = true;
        vertexVelocities.neverGPU = true;
        vertexMasses.neverGPU = true;
        vertexNeighbors.neverGPU = true;
        vertexCellNeighbors.neverGPU = true;
        neighborNum.neverGPU = true;
        neighbors.neverGPU = true;
        cellVertexNum.neverGPU = true;
        vertexForces.neverGPU = true;
        cellForces.neverGPU = true;
        cellType.neverGPU = true;
        Moduli.neverGPU = true;
        AreaPeri.neverGPU = true;
        AreaPeriPreferences.neverGPU = true;
        cellVertices.neverGPU = true;
        voroCur.neverGPU = true;
        voroLastNext.neverGPU = true;
        displacements.neverGPU = true;
        };

    //setting cell positions randomly also auto-generates a square box with L = sqrt(Ncells)
    setCellPositionsRandomly();
    AreaPeri.resize(Ncells);
    cellForces.resize(Ncells);
    setCellPreferencesUniform(1.0,3.8);
    setModuliUniform(1.0,1.0);
    setCellTypeUniform(0);
    cellMasses.resize(Ncells);
    cellVelocities.resize(Ncells);
    vector<double> masses(Ncells,1.0);
    fillGPUArrayWithVector(masses,cellMasses);
    vector<double2> velocities(Ncells,make_double2(0.0,0.0));
    fillGPUArrayWithVector(velocities,cellVelocities);

    vertexForces.resize(Nvertices);
    };

/*!
Generically believe that cells in 2D have a notion of a preferred area and perimeter
*/
void Simple2DCell::setCellPreferencesUniform(double A0, double P0)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<double2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = A0;
        h_p.data[ii].y = P0;
        };
    };

/*!
choose a target p_0, randomly distribute target a_0s (a las pica ciamarra and paoluzzi et al paper
*/
void Simple2DCell::setCellPreferencesWithRandomAreas(double p0, double aMin,double aMax)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<double2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double a = noise.getRealUniform(aMin, aMax);
        h_p.data[ii].x = a;
        h_p.data[ii].y = p0 * sqrt(a);
        };
    };

/*!
Set the Area and Perimeter preferences to the input vector
*/
void Simple2DCell::setCellPreferences(vector<double2> &APPref)
    {
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<double2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = APPref[ii].x;
        h_p.data[ii].y = APPref[ii].y;
        };
    };

/*!
Simply call either the CPU or GPU routine in the current or derived model
*/
void Simple2DCell::computeGeometry()
    {
    if(GPUcompute)
        computeGeometryGPU();
    else
        computeGeometryCPU();
    }

/*!
Resize the box so that the new dimensions are Lx by Ly, and rescale the positions of existing cells
*/
void Simple2DCell::setRectangularUnitCell(double Lx, double Ly)
    {
    double bxx,bxy,byy,byx;
    Box->getBoxDims(bxx,bxy,byx,byy);
    Box->setSquare(Lx,Ly);

    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_p.data[ii].x = h_p.data[ii].x*Lx/bxx;
        h_p.data[ii].y = h_p.data[ii].y*Ly/byy;
        };
    }

/*!
Resize the box so that every cell has, on average, area = 1, and place cells via either a simple,
reproducible RNG or a non-reproducible RNG
*/
void Simple2DCell::setCellPositionsRandomly()
    {
    cellPositions.resize(Ncells);
    double boxsize = sqrt((double)Ncells);
    Box->setSquare(boxsize,boxsize);
    noise.Reproducible = Reproducible;

    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double x = noise.getRealUniform(0.0,boxsize);
        double y = noise.getRealUniform(0.0,boxsize);
        h_p.data[ii].x = x;
        h_p.data[ii].y = y;
        };
    };

/*!
Does not update any other lists -- it is the user's responsibility to maintain topology, etc, when
using this function.
*/
void Simple2DCell::setCellPositions(vector<double2> newCellPositions)
    {
    Ncells = newCellPositions.size();
    if(cellPositions.getNumElements() != Ncells) cellPositions.resize(Ncells);
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        h_p.data[ii] = newCellPositions[ii];
    }

/*!
Does not update any other lists -- it is the user's responsibility to maintain topology, etc, when
using this function.
*/
void Simple2DCell::setVertexPositions(vector<double2> newVertexPositions)
    {
    Nvertices = newVertexPositions.size();
    if(vertexPositions.getNumElements() != Nvertices) vertexPositions.resize(Nvertices);
    ArrayHandle<double2> h_v(vertexPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        h_v.data[ii] = newVertexPositions[ii];
    }


/*!
set all cell K_A, K_P preferences to uniform values.
PLEASE NOTE that as an optimization this data is not actually used at the moment,
but the code could be trivially altered to use this
*/
void Simple2DCell::setModuliUniform(double newKA, double newKP)
    {
    KA=newKA;
    KP=newKP;
    Moduli.resize(Ncells);
    ArrayHandle<double2> h_m(Moduli,access_location::host,access_mode::overwrite);
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
    cellType.resize(Ncells);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ct.data[ii] = i;
        };
    };

/*!
Set the vertex velocities by drawing from a Maxwell-Boltzmann distribution, and then make sure there is no
net momentum. The return value is the total kinetic energy.
 */
double Simple2DCell::setVertexVelocitiesMaxwellBoltzmann(double T)
    {
    noise.Reproducible = Reproducible;
    ArrayHandle<double> h_cm(vertexMasses,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(vertexVelocities,access_location::host,access_mode::overwrite);
    double2 P = make_double2(0.0,0.0);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        double2 vi;
        vi.x = noise.getRealNormal(0.0,sqrt(T/h_cm.data[ii]));
        vi.y = noise.getRealNormal(0.0,sqrt(T/h_cm.data[ii]));
        h_v.data[ii] = vi;
        P = P+h_cm.data[ii]*vi;
        };
    //remove excess momentum, calculate the KE
    double KE = 0.0;
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        h_v.data[ii] = h_v.data[ii] + (-1.0/(Ncells*h_cm.data[ii]))*P;
        KE += 0.5*h_cm.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    return KE;
    };

/*!
Set the cell velocities by drawing from a Maxwell-Boltzmann distribution, and then make sure there is no
net momentum. The return value is the total kinetic energy
 */
double Simple2DCell::setCellVelocitiesMaxwellBoltzmann(double T)
    {
    noise.Reproducible = Reproducible;
    ArrayHandle<double> h_cm(cellMasses,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(cellVelocities,access_location::host,access_mode::overwrite);
    double2 P = make_double2(0.0,0.0);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double2 vi;
        vi.x = noise.getRealNormal(0.0,sqrt(T/h_cm.data[ii]));
        vi.y = noise.getRealNormal(0.0,sqrt(T/h_cm.data[ii]));
        h_v.data[ii] = vi;
        P = P+h_cm.data[ii]*vi;
        };
    //remove excess momentum, calculate the KE
    double KE = 0.0;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_v.data[ii] = h_v.data[ii] + (-1.0/(Ncells*h_cm.data[ii]))*P;
        KE += 0.5*h_cm.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    return KE;
    };

/*!
 \param types a vector of integers that the cell types will be set to
 */
void Simple2DCell::setCellType(vector<int> &types)
    {
    cellType.resize(Ncells);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
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
void Simple2DCell::reIndexCellArray(GPUArray<double2> &array)
    {
    GPUArray<double2> TEMP = array;
    ArrayHandle<double2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
Re-indexes GPUarrays of doubles
*/
void Simple2DCell::reIndexCellArray(GPUArray<double> &array)
    {
    GPUArray<double> TEMP = array;
    ArrayHandle<double> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double> ar(array,access_location::host,access_mode::readwrite);
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
void Simple2DCell::reIndexVertexArray(GPUArray<double2> &array)
    {
    GPUArray<double2> TEMP = array;
    ArrayHandle<double2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Nvertices; ++ii)
        {
        ar.data[ii] = temp.data[ittVertex[ii]];
        };
    };

void Simple2DCell::reIndexVertexArray(GPUArray<double> &array)
    {
    GPUArray<double> TEMP = array;
    ArrayHandle<double> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double> ar(array,access_location::host,access_mode::readwrite);
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
    HilbertSorter hs(*(Box));

    vector<pair<int,int> > idxCellSorter(Ncells);

    //sort points by Hilbert Curve location
    ArrayHandle<double2> h_p(cellPositions,access_location::host, access_mode::readwrite);
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
    reIndexCellArray(AreaPeri);
    reIndexCellArray(cellType);
    reIndexCellArray(cellVelocities);
    reIndexCellArray(cellMasses);
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
    HilbertSorter hs(*(Box));

    vector<pair<int,int> > idxSorterVertex(Nvertices);

    //sort points by Hilbert Curve location
    ArrayHandle<double2> h_p(vertexPositions,access_location::host, access_mode::readwrite);
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
    reIndexCellArray(vertexVelocities);
    reIndexCellArray(vertexMasses);

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
    reIndexCellArray(cellType);
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
P_ab = \sum m_i v_{ib}v_{ia}
*/
double4 Simple2DCell::computeKineticPressure()
    {
    int Ndof = getNumberOfDegreesOfFreedom();
    double4 ans; ans.x = 0.0; ans.y=0.0;ans.z=0;ans.w=0.0;
    ArrayHandle<double> h_m(returnMasses());
    ArrayHandle<double2> h_v(returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        {
        double  m = h_m.data[ii];
        double2 v = h_v.data[ii];
        ans.x += m*v.x*v.x;
        ans.y += m*v.y*v.x;
        ans.z += m*v.x*v.y;
        ans.w += m*v.y*v.y;
        };
    double b1,b2,b3,b4;
    Box->getBoxDims(b1,b2,b3,b4);
    double area = b1*b4;
    ans.x = ans.x / area;
    ans.y = ans.y / area;
    ans.z = ans.z / area;
    ans.w = ans.w / area;
    return ans;
    };

/*!
E = \sum 0.5*m_i v_i^2
*/
double Simple2DCell::computeKineticEnergy()
    {
    int Ndof = getNumberOfDegreesOfFreedom();
    double ans = 0.0;
    ArrayHandle<double> h_m(returnMasses());
    ArrayHandle<double2> h_v(returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        {
        double  m = h_m.data[ii];
        double2 v = h_v.data[ii];
        ans += 0.5*m*(v.x*v.x+v.y*v.y);
        };
    return ans;
    };

/*!
a utility/testing function...output the currently computed mean net force to screen.
\param verbose if true also print out the force on each cell
*/
void Simple2DCell::reportMeanCellForce(bool verbose)
    {
    ArrayHandle<double2> h_f(cellForces,access_location::host,access_mode::read);
    ArrayHandle<double2> p(cellPositions,access_location::host,access_mode::read);
    double fx = 0.0;
    double fy = 0.0;
    double min = 10000;
    double max = -10000;
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
Returns the mean value of the perimeter
*/
double Simple2DCell::reportMeanP()
    {
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    double P = 0.0;
    double ans = 0.0;
    for (int i = 0; i < Ncells; ++i)
        {
        P = h_AP.data[i].y;
        ans += P ;
        };
    return ans/(double)Ncells;
    };

/*!
Returns the mean value of the shape parameter:
*/
double Simple2DCell::reportq()
    {
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    double A = 0.0;
    double P = 0.0;
    double q = 0.0;
    for (int i = 0; i < Ncells; ++i)
        {
        A = h_AP.data[i].x;
        P = h_AP.data[i].y;
//    printf("%f\t",P/sqrt(A));
        q += P / sqrt(A);
        };
    return q/(double)Ncells;
    };

/*!
Returns the variance of the shape parameter:
*/
double Simple2DCell::reportVarq()
    {
    double meanQ = reportq();
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    double qtemp = 0.0;
    double var = 0.0;
    for (int i = 0; i < Ncells; ++i)
        {
        double A = h_AP.data[i].x;
        double P = h_AP.data[i].y;
        qtemp = P / sqrt(A);
        var += (qtemp-meanQ)*(qtemp-meanQ);
        };
    return var/(double)Ncells;
    };

/*!
Returns the variance of the A and P for the system:
*/
double2 Simple2DCell::reportVarAP()
    {
    double meanA = 0;
    double meanP = 0;
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    for (int i = 0; i < Ncells; ++i)
        {
        double A = h_AP.data[i].x;
        double P = h_AP.data[i].y;
        meanA += A;
        meanP += P;
        };
    meanA = meanA /(double)Ncells;
    meanP = meanP /(double)Ncells;

    double2 var;
    var.x=0.0; var.y=0.0;
    for (int i = 0; i < Ncells; ++i)
        {
        double A = h_AP.data[i].x;
        double P = h_AP.data[i].y;
        var.x += (A-meanA)*(A-meanA);
        var.y += (P-meanP)*(P-meanP);
        };
    var.x = var.x /(double)Ncells;
    var.y = var.y /(double)Ncells;

    return var;
    };

/*!
 * When beggars die, there are no comets seen;
 * The heavens themselves blaze forth the death of princes...
 Which are your cells?
This function supports removing a single cell from the simulation, which requires re-indexing
and relabeling the data structures in the simulation.
\post Simple2DCell data vectors are one element shorter, and cells with a higher index than "cellIndex" get re-labeled down by one.
*/
void Simple2DCell::cellDeath(int cellIndex)
    {
    Ncells -= 1;
    forcesUpToDate=false;

    //reset the spatial sorting vectors...
    itt.resize(Ncells);
    tti.resize(Ncells);
    vector<int> newTagToIdx(Ncells);
    vector<int> newIdxToTag(Ncells);
    int loopIndex = 0;
    for (int ii = 0; ii < Ncells+1;++ii)
        {
        int pIdx = tagToIdx[ii]; //pIdx is the current position of the cell that was originally ii
        if(pIdx != cellIndex)
            {
            if (pIdx >= cellIndex) pIdx = pIdx-1;
            newTagToIdx[loopIndex] = pIdx;
            loopIndex +=1;
            };
        };
    for (int ii = 0; ii < Ncells; ++ii)
        newIdxToTag[newTagToIdx[ii]] = ii;
    tagToIdx = newTagToIdx;
    idxToTag = newIdxToTag;

    //AreaPeri will have its values updated in a geometry routine... just change the length
    AreaPeri.resize(Ncells);
    //use the GPUArray removal mechanism to get rid of the correct data
    removeGPUArrayElement(AreaPeriPreferences,cellIndex);
    removeGPUArrayElement(Moduli,cellIndex);
    removeGPUArrayElement(cellMasses,cellIndex);
    removeGPUArrayElement(cellVelocities,cellIndex);
    removeGPUArrayElement(cellType,cellIndex);
    removeGPUArrayElement(cellPositions,cellIndex);
    };

/*!
This function supports cellDivisions, updating data structures in Simple2DCell
This function will grow the cell lists by 1 and assign the new cell
(the last element of those arrays) the values of the cell given by parameters[0]
Note that dParams does nothing by default, but allows more general virtual functions to be defined
downstream (used in the Voronoi branch)
 */
void Simple2DCell::cellDivision(const vector<int> &parameters, const vector<double> &dParams)
    {
    forcesUpToDate=false;
    Ncells += 1;
    n_idx = Index2D(vertexMax,Ncells);
    int cellIdx = parameters[0];

    //additions to the spatial sorting vectors...
    itt.push_back(Ncells-1);
    tti.push_back(Ncells-1);
    tagToIdx.push_back(Ncells-1);
    idxToTag.push_back(Ncells-1);

    //AreaPeri will have its values updated in a geometry routine... just change the length
    AreaPeri.resize(Ncells);

    //use the copy and grow mechanism where we need to actually set values
    growGPUArray(AreaPeriPreferences,1); //(nc)
    growGPUArray(Moduli,1);
    growGPUArray(cellMasses,1);
    growGPUArray(cellVelocities,1);
    growGPUArray(cellType,1);
    growGPUArray(cellPositions,1);

        {//arrayhandle scope
        ArrayHandle<double2> h_APP(AreaPeriPreferences); h_APP.data[Ncells-1] = h_APP.data[cellIdx];
        ArrayHandle<double2> h_Mod(Moduli); h_Mod.data[Ncells-1] = h_Mod.data[cellIdx];
        ArrayHandle<int> h_ct(cellType); h_ct.data[Ncells-1] = h_ct.data[cellIdx];
        ArrayHandle<double> h_cm(cellMasses);  h_cm.data[Ncells-1] = h_cm.data[cellIdx];
        ArrayHandle<double2> h_v(cellVelocities); h_v.data[Ncells-1] = make_double2(0.0,0.0);
        };
    };
