#define ENABLE_CUDA

#include "vertexModelBase.h"
#include "vertexModelBase.cuh"
#include "voronoi2d.h"
/*! \file vertexModelBase.cpp */

/*!
move vertices according to an inpute GPUarray
*/
void vertexModelBase::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements)
    {
    //handle things either on the GPU or CPU
    if (GPUcompute)
        {
        ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<Dscalar2> d_v(vertexPositions,access_location::device,access_mode::readwrite);
        gpu_vm_displace(d_v.data,
                         d_disp.data,
                         Box,
                         Nvertices);
        }
    else
        {
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::read);
        ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::readwrite);
        for (int i = 0; i < Nvertices; ++i)
            {
            h_v.data[i].x += h_disp.data[i].x;
            h_v.data[i].y += h_disp.data[i].y;
            Box.putInBoxReal(h_v.data[i]);
            };
        };
    };

/*!
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures
(the vertices)...compute the area and perimeter of the cells
*/
void vertexModelBase::computeGeometryCPU()
    {
    ArrayHandle<Dscalar2> h_v(vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellVertices,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);

    //compute the geometry for each cell
    for (int i = 0; i < Ncells; ++i)
        {
        int neighs = h_nn.data[i];
//      Define the vertices of a cell relative to some (any) of its verties to take care of periodic boundaries
        Dscalar2 cellPos = h_v.data[h_n.data[n_idx(neighs-2,i)]];
        Dscalar2 vlast, vcur,vnext;
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        //compute the vertex position relative to the cell position
        vlast.x=0.;vlast.y=0.0;
        int vidx = h_n.data[n_idx(neighs-1,i)];
        Box.minDist(h_v.data[vidx],cellPos,vcur);
        for (int nn = 0; nn < neighs; ++nn)
            {
            //for easy force calculation, save the current, last, and next vertex position in the approprate spot.
            int forceSetIdx= -1;
            for (int ff = 0; ff < 3; ++ff)
                if(h_vcn.data[3*vidx+ff]==i)
                    forceSetIdx = 3*vidx+ff;

            vidx = h_n.data[n_idx(nn,i)];
            Box.minDist(h_v.data[vidx],cellPos,vnext);

            //contribution to cell's area is
            // 0.5* (vcur.x+vnext.x)*(vnext.y-vcur.y)
            Varea += SignedPolygonAreaPart(vcur,vnext);
            Dscalar dx = vcur.x-vnext.x;
            Dscalar dy = vcur.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            //save vertex positions in a convenient form
            h_vc.data[forceSetIdx] = vcur;
            h_vln.data[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
            //advance the loop
            vlast = vcur;
            vcur = vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

/*!
Very similar to the function in Voronoi2d.cpp, but optimized since we already have some data structures (the vertices)
*/
void vertexModelBase::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_v(vertexPositions,      access_location::device,access_mode::read);
    ArrayHandle<int>      d_cvn(cellVertexNum,       access_location::device,access_mode::read);
    ArrayHandle<int>      d_cv(cellVertices,         access_location::device,access_mode::read);
    ArrayHandle<int>      d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,             access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,       access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,            access_location::device,access_mode::overwrite);

    gpu_vm_geometry(
                    d_v.data,
                    d_cvn.data,
                    d_cv.data,
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    Ncells,n_idx,Box);
    };

/*!
One would prefer the cell position to be defined as the centroid, requiring an additional computation of the cell area.
This may be implemented some day, but for now we define the cell position as the straight average of the vertex positions.
This isn't really used much, anyway, so update this only when the functionality becomes needed
*/
void vertexModelBase::getCellPositionsCPU()
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
            Box.minDist(h_v.data[vidx],baseVertex,vertex);
            pos.x += vertex.x;
            pos.y += vertex.y;
            };
        pos.x /= neighs;
        pos.y /= neighs;
        pos.x += baseVertex.x;
        pos.y += baseVertex.y;
        Box.putInBoxReal(pos);
        h_p.data[cell] = pos;
        };
    };

/*!
Repeat the above calculation of "cell positions", but on the GPU
*/
void vertexModelBase::getCellPositionsGPU()
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
                               Box);
    };

/*!
Trigger a cell division event, which involves some laborious re-indexing of various data structures.
This simple version of cell division will take a cell and two specified vertices. The edges emanating
clockwise from each of the two vertices will gain a new vertex in the middle of those edges. A new cell is formed by connecting those two new vertices together.
The vector of "parameters" here should be three integers:
parameters[0] = the index of the cell to undergo a division event
parameters[1] = the first vertex to gain a new (clockwise) vertex neighbor. 
parameters[2] = the second .....
The two vertex numbers should be between 0 and celLVertexNum[parameters[0]], respectively, NOT the
indices of the vertices being targeted
*/
void vertexModelBase::cellDivision(vector<int> &parameters)
    {
    };
