#define ENABLE_CUDA

#include "simpleVertexModelBase.h"
#include "simpleVertexModelBase.cuh"
/*! \file simpleVertexModelBase.cpp */


/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void simpleVertexModelBase::initializeSimpleVertexModelBase(int n)
    {
    //set number of cells, and call initializer chain
    Ncells=n;
    initializeSimple2DActiveCell(Ncells);

    //initializes per-cell lists
    initializeCellSorting();
    };


/*!
move vertices according to an input GPUarray
*/
void simpleVertexModelBase::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements,Dscalar scale)
    {
    forcesUpToDate = false;
    //handle things either on the GPU or CPU
    if (GPUcompute)
        {
        ArrayHandle<Dscalar2> d_d(displacements,access_location::device,access_mode::read);
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
        if (scale == 1.)
            gpu_move_degrees_of_freedom(d_v.data,d_d.data,Nvertices,*(Box));
        else
            gpu_move_degrees_of_freedom(d_v.data,d_d.data,scale,Nvertices,*(Box));
        }
    else
        {
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
        if(scale ==1.)
            {
            for (int i = 0; i < Nvertices; ++i)
                {
                h_v.data[i].x += h_disp.data[i].x;
                h_v.data[i].y += h_disp.data[i].y;
                Box->putInBoxReal(h_v.data[i]);
                };
            }
        else
            {
            for (int i = 0; i < Nvertices; ++i)
                {
                h_v.data[i].x += scale*h_disp.data[i].x;
                h_v.data[i].y += scale*h_disp.data[i].y;
                Box->putInBoxReal(h_v.data[i]);
                };
            }
        };
    };

/*!
 *When sortPeriod < 0 this routine does not get called
 \post vertices are re-ordered according to a Hilbert sorting scheme, cells are reordered according
 to what vertices they are near, and all data structures are updated
 */
void simpleVertexModelBase::spatialSorting()
    {
    //the base vertex model class doesn't need to change any other unusual data structures at the moment
    spatiallySortVerticesAndCellActivity();
    reIndexVertexArray(vertexMasses);
    reIndexVertexArray(vertexVelocities);
    };

/*!
when a transition increases the maximum number of vertices around any cell in the system,
call this function first to copy over the cellVertices structure into a larger array
 */
void simpleVertexModelBase::growCellVerticesList(int newVertexMax)
    {
    cout << "maximum number of vertices per cell grew from " <<vertexMax << " to " << newVertexMax << endl;
    vertexMax = newVertexMax+1;
    Index2D old_idx = n_idx;
    n_idx = Index2D(vertexMax,Ncells);

    GPUArray<int> newCellVertices;
    newCellVertices.resize(vertexMax*Ncells);
    {//scope for array handles
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n_old(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(newCellVertices,access_location::host,access_mode::readwrite);

    for(int cell = 0; cell < Ncells; ++cell)
        {
        int neighs = h_nn.data[cell];
        for (int n = 0; n < neighs; ++n)
            {
            h_n.data[n_idx(n,cell)] = h_n_old.data[old_idx(n,cell)];
            };
        };
    };//scope for array handles
    cellVertices.resize(vertexMax*Ncells);
    cellVertices.swap(newCellVertices);
    };

/*!
This function fills the "cellPositions" GPUArray with the centroid of every cell. Does not assume
that the area in the AreaPeri array is current. This function just calls the CPU or GPU routine, as determined by the GPUcompute flag
*/
void simpleVertexModelBase::getCellCentroids()
    {
    if(GPUcompute)
        getCellCentroidsGPU();
    else
        getCellCentroidsCPU();
    };

/*!
GPU computation of the centroid of every cell
*/
void simpleVertexModelBase::getCellCentroidsGPU()
    {
    printf("getCellCentroidsGPU() function not currently functional...Very sorry\n");
    throw std::exception();
    };

/*!
CPU computation of the centroid of every cell
*/
void simpleVertexModelBase::getCellCentroidsCPU()
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);

    Dscalar2 zero = make_Dscalar2(0.0,0.0);
    Dscalar2 baseVertex;
    for (int cell = 0; cell < Ncells; ++cell)
        {
        //for convenience, for each cell we will make a vector of the vertices of the cell relative to vertex 0
        //the vector will be of length (vertices+1), and the first and last entry will be zero.
        baseVertex = h_v.data[h_n.data[n_idx(0,cell)]];
        int neighs = h_nn.data[cell];
        vector<Dscalar2> vertices(neighs+1,zero);
        for (int vv = 1; vv < neighs; ++vv)
            {
            int vidx = h_n.data[n_idx(vv,cell)];
            Box->minDist(h_v.data[vidx],baseVertex,vertices[vv]);
            };
        //compute the area and the sums for the centroids
        Dscalar Area = 0.0;
        Dscalar2 centroid = zero;
        for (int vv = 0; vv < neighs; ++vv)
            {
            Area += (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            centroid.x += (vertices[vv].x+vertices[vv+1].x) * (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            centroid.y += (vertices[vv].y+vertices[vv+1].y) * (vertices[vv].x*vertices[vv+1].y - vertices[vv+1].x*vertices[vv].y);
            };
        Area = 0.5*Area;
        centroid.x = centroid.x / (6.0*Area) + baseVertex.x;
        centroid.y = centroid.y / (6.0*Area) + baseVertex.y;
        Box->putInBoxReal(centroid);
        h_p.data[cell] = centroid;
        };
    };

/*!
This function fills the "cellPositions" GPUArray with the mean position of the vertices of each cell.
This function just calls the CPU or GPU routine, as determined by the GPUcompute flag
*/
void simpleVertexModelBase::getCellPositions()
    {
    if(GPUcompute)
        getCellPositionsGPU();
    else
        getCellPositionsCPU();
    };
/*!
One would prefer the cell position to be defined as the centroid, requiring an additional computation of the cell area.
This may be implemented some day, but for now we define the cell position as the straight average of the vertex positions.
This isn't really used much, anyway, so update this only when the functionality becomes needed
*/
void simpleVertexModelBase::getCellPositionsCPU()
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);

    Dscalar2 vertex,baseVertex,pos;
    for (int cell = 0; cell < Ncells; ++cell)
        {
        baseVertex = h_v.data[h_n.data[n_idx(0,cell)]];
        int neighs = h_nn.data[cell];
        pos.x=0.0;pos.y=0.0;
        //compute the vertex position relative to the cell position
        for (int n = 1; n < neighs; ++n)
            {
            int vidx = h_n.data[n_idx(n,cell)];
            Box->minDist(h_v.data[vidx],baseVertex,vertex);
            pos.x += vertex.x;
            pos.y += vertex.y;
            };
        pos.x /= neighs;
        pos.y /= neighs;
        pos.x += baseVertex.x;
        pos.y += baseVertex.y;
        Box->putInBoxReal(pos);
        h_p.data[cell] = pos;
        };
    };

/*!
Repeat the above calculation of "cell positions", but on the GPU
*/
void simpleVertexModelBase::getCellPositionsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_cv(cellVertices,access_location::device,access_mode::read);

    gpu_vm_get_cell_positions(d_p.data,
                               d_v.data,
                               d_cvn.data,
                               d_cv.data,
                               Ncells,
                               n_idx,
                               *(Box));
    };
