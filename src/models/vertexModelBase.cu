#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "vertexModelBase.cuh"

/** \file vertexModelBase.cu
    * Defines kernel callers and kernels for GPU calculations of vertex models
*/

/*!
    \addtogroup vmKernels
    @{
*/

/*!
  Since the cells are NOT guaranteed to be convex, the area of the cell must take into account any
  self-intersections. The strategy is the same as in the CPU branch.
  */
__global__ void vm_geometry_kernel(
                                   const Dscalar2* __restrict__ d_vertexPositions,
                                   const int*  __restrict__ d_cellVertexNum,
                                   const int*  __restrict__ d_cellVertices,
                                   const int*  __restrict__ d_vertexCellNeighbors,
                                   Dscalar2*  __restrict__ d_voroCur,
                                   Dscalar4*  __restrict__ d_voroLastNext,
                                   Dscalar2*  __restrict__ d_AreaPerimeter,
                                   int N,
                                   Index2D n_idx,
                                   gpubox Box
                                    )
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int neighs = d_cellVertexNum[idx];
    //Define the vertices of a cell relative to some (any one ) of its vertices to take care of periodic BCs
    Dscalar2 cellPos = d_vertexPositions[ d_cellVertices[n_idx(neighs-2,idx)]];
    Dscalar2 vlast, vcur,vnext;
    Dscalar Varea = 0.0;
    Dscalar Vperi = 0.0;

    vlast.x = 0.0; vlast.y=0.0;
    int vidx = d_cellVertices[n_idx(neighs-1,idx)];
    Box.minDist(d_vertexPositions[vidx],cellPos,vcur);
    for (int nn = 0; nn < neighs; ++nn)
        {
        //for easy force calculation, save the current, last, and next voronoi vertex position
        //in the approprate spot.
        int forceSetIdx = -1;
        for (int ff = 0; ff < 3; ++ff)
            {
            if(forceSetIdx != -1) continue;
            if(d_vertexCellNeighbors[3*vidx+ff]==idx)
                forceSetIdx = 3*vidx+ff;
            };

        vidx = d_cellVertices[n_idx(nn,idx)];
        Box.minDist(d_vertexPositions[vidx],cellPos,vnext);

        //compute area contribution. It is
        // 0.5 * (vcur.x+vnext.x)*(vnext.y-vcur.y)
        Varea += SignedPolygonAreaPart(vcur,vnext);
        Dscalar dx = vcur.x-vnext.x;
        Dscalar dy = vcur.y-vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        //save voronoi positions in a convenient form
        d_voroCur[forceSetIdx] = vcur;
        d_voroLastNext[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
        //advance the loop
        vlast = vcur;
        vcur = vnext;
        };
    d_AreaPerimeter[idx].x=Varea;
    d_AreaPerimeter[idx].y=Vperi;
    };

/*!
  This function is being deprecated, but is still useful for calculating, e.g. the mean-squared
displacement of the cells without transferring data back to the host
*/
__global__ void vm_get_cell_positions_kernel(Dscalar2* d_cellPositions,
                                              Dscalar2* d_vertexPositions,
                                              int    * d_nn,
                                              int    * d_n,
                                              int N,
                                              Index2D n_idx,
                                              gpubox Box)
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    Dscalar2 vertex, pos, baseVertex;
    pos.x=0.0;pos.y=0.0;
    baseVertex = d_vertexPositions[ d_n[n_idx(0,idx)] ];
    int neighs = d_nn[idx];
    for (int n = 1; n < neighs; ++n)
        {
        Box.minDist(d_vertexPositions[ d_n[n_idx(n,idx)] ],baseVertex,vertex);
        pos.x += vertex.x;
        pos.y += vertex.y;
        };
    pos.x /= neighs;
    pos.y /= neighs;
    pos.x += baseVertex.x;
    pos.y += baseVertex.y;
    Box.putInBoxReal(pos);
    d_cellPositions[idx] = pos;
    };

/*!
  Run through every pair of vertices (once), see if any T1 transitions should be done,
  and see if the cell-vertex list needs to grow
  */
__global__ void vm_simple_T1_test_kernel(Dscalar2* d_vertexPositions,
                                        int      *d_vertexNeighbors,
                                        int      *d_vertexEdgeFlips,
                                        int      *d_vertexCellNeighbors,
                                        int      *d_cellVertexNum,
                                        gpubox   Box,
                                        Dscalar  T1THRESHOLD,
                                        int      NvTimes3,
                                        int      vertexMax,
                                        int      *d_grow)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= NvTimes3)
        return;
    int vertex1 = idx/3;
    int vertex2 = d_vertexNeighbors[idx];
    Dscalar2 edge;
    if(vertex1 < vertex2)
        {
        Box.minDist(d_vertexPositions[vertex1],d_vertexPositions[vertex2],edge);
        if(norm(edge) < T1THRESHOLD)
            {
            d_vertexEdgeFlips[idx]=1;


            //test the number of neighbors of the cells connected to v1 and v2 to see if the
            //cell list should grow. This is kind of slow, and I wish I could optimize it away,
            //or at least not test for it during every time step. The latter seems pretty doable.
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex1+2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2+1]] == vertexMax)
                d_grow[0] = 1;
            if(d_cellVertexNum[d_vertexCellNeighbors[3*vertex2+2]] == vertexMax)
                d_grow[0] = 1;
            }
        else
            d_vertexEdgeFlips[idx]=0;
        }
    else
        d_vertexEdgeFlips[idx] = 0;
    };

/*!
  There will be severe topology mismatches if a cell is involved in more than one T1 transition
  simultaneously (due to incoherent updates of the cellVertices structure). So, go through the
  current list of edges that are marked to take part in a T1 transition and select one edge per
  cell to be flipped on this trip through the functions.
  */
__global__ void vm_one_T1_per_cell_per_vertex_kernel(
                                        int* __restrict__ d_vertexEdgeFlips,
                                        int* __restrict__ d_vertexEdgeFlipsCurrent,
                                        const int* __restrict__ d_vertexNeighbors,
                                        const int* __restrict__ d_vertexCellNeighbors,
                                        const int* __restrict__ d_cellVertexNum,
                                        const int * __restrict__ d_cellVertices,
                                        int *d_finishedFlippingEdges,
                                        int *d_cellEdgeFlips,
                                        int4 *d_cellSets,
                                        Index2D n_idx,
                                        int Ncells)
    {
    unsigned int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= Ncells)
        return;

    //look through every vertex of the cell
    int cneigh = d_cellVertexNum[cell];
    int vertex;
    bool flipFound = false;
    bool moreFlipsFound = false;
    for (int cc = 0; cc < cneigh; ++cc)
        {
        vertex = d_cellVertices[n_idx(cc,cell)];
        //what are the other cells attached to this vertex? For correctness, only one cell should
        //own each vertex here. For simplicity, only the lowest-indexed cell gets to do any work.
        int c1,c2,c3,c4;
        c1 = d_vertexCellNeighbors[3*vertex];
        c2 = d_vertexCellNeighbors[3*vertex+1];
        c3 = d_vertexCellNeighbors[3*vertex+2];
        if(c1 < cell || c2 < cell || c3 < cell)
            continue;

        for (int idx = 3*vertex; idx < 3*vertex+3; ++idx)
            {
            if(d_vertexEdgeFlips[idx] == 1)
                {
                int vertex2 = d_vertexNeighbors[idx];
                int ctest;
                for (int ff = 0; ff < 3; ++ff)
                    {
                    ctest = d_vertexCellNeighbors[3*vertex2+ff];
                    if(ctest != c1 && ctest != c2 && ctest != c3)
                    c4=ctest;
                    }

                if (flipFound)
                    {
                    moreFlipsFound = true;
                    break;
                    }
                //check if the cells have been reserved; if not reserve them
                int cc1 = atomicExch(&(d_cellEdgeFlips[c1]),1);
                int cc2 = atomicExch(&(d_cellEdgeFlips[c2]),1);
                int cc3 = atomicExch(&(d_cellEdgeFlips[c3]),1);
                int cc4 = atomicExch(&(d_cellEdgeFlips[c4]),1);
                flipFound = true;
                if(cc1 ==0 && cc2 ==0 &&cc3==0&&cc4==0)
                    {
//                printf("(%i,%i,%i,%i)\t(%i,%i)\n",c1,c2,c3,c4,vertex,vertex2);
                    atomicExch(&d_vertexEdgeFlipsCurrent[idx],1);
                    atomicExch(&d_vertexEdgeFlips[idx],0);
                    int4 cs;cs.x=c1;cs.y=c2;cs.z=c3;cs.w=c4;
                    d_cellSets[idx] = cs;
                    };
                }
            };
        };
    if (flipFound)
        {
        d_finishedFlippingEdges[0] = 1;
        if(moreFlipsFound)
            d_finishedFlippingEdges[1] = 1;
        };
    };

/*!
  Flip any edge labeled for re-wiring in the vertexEdgeFlipsCurrent list
  */
__global__ void vm_flip_edges_kernel(int* d_vertexEdgeFlipsCurrent,
                                      Dscalar2 *d_vertexPositions,
                                      int      *d_vertexNeighbors,
                                      int      *d_vertexCellNeighbors,
                                      int      *d_cellVertexNum,
                                      int      *d_cellVertices,
                                      int      *d_cellEdgeFlips,
                                      int4 *d_cellSets,
                                      gpubox   Box,
                                      Index2D  n_idx,
                                      int      NvTimes3)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //return if the index is out of bounds or if the edge isn't marked for flipping
    if (idx >= NvTimes3 || d_vertexEdgeFlipsCurrent[idx] == 0)
        return;
    //identify the vertices and reset the flag
    int vertex1 = idx/3;
    int vertex2 = d_vertexNeighbors[idx];
    d_vertexEdgeFlipsCurrent[idx] = 0;

    //first, identify the cell and vertex set involved...
    int4 cellSet;cellSet.x=-1;cellSet.y=-1;cellSet.z=-1;cellSet.w=-1;
    //int4 vertexSet;
    int2 vertexSet;//vertexSet.x = "b", vertexSet.y = "a"
    /*
    The following is fairly terrible GPU code, and should be considered for refactoring
    */
    int4 cells = d_cellSets[idx];
    int cell1,cell2,cell3;
    int vlast, vcur, vnext, cneigh;
    cell1 = cells.x;
    cell2 = cells.y;
    cell3 = cells.z;
    cellSet.w = cells.w;

    //classify cell1
    cneigh = d_cellVertexNum[cell1];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell1) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell1) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell1)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell1;
    else if(vnext == vertex2)
        cellSet.z = cell1;
    else
        {
        cellSet.y = cell1;
        };

    //classify cell2
    cneigh = d_cellVertexNum[cell2];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell2) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell2) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell2)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell2;
    else if(vnext == vertex2)
        cellSet.z = cell2;
    else
        {
        cellSet.y = cell2;
        };
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell1)];
        }
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell2)];
        }
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell3)];
        }

    //classify cell3
    cneigh = d_cellVertexNum[cell3];
    vlast = d_cellVertices[ n_idx(cneigh-2,cell3) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cell3) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cell3)];
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
    if(vlast == vertex2)
        cellSet.x = cell3;
    else if(vnext == vertex2)
        cellSet.z = cell3;
    else
        {
        cellSet.y = cell3;
        };

    //get the vertexSet by examining cells j and l
    cneigh = d_cellVertexNum[cellSet.y];
    vlast = d_cellVertices[ n_idx(cneigh-2,cellSet.y) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cellSet.y) ];
//    printf("\n");
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cellSet.y)];
//        if(vertex1<10)printf("\t%i (%i,%i,%i)\n",cneigh,vlast,vcur,vnext);
        if(vcur == vertex1) break;
        vlast = vcur;
        vcur = vnext;
        };
//    printf("\n");
    vertexSet.x=vnext;
    cneigh = d_cellVertexNum[cellSet.w];
    vlast = d_cellVertices[ n_idx(cneigh-2,cellSet.w) ];
    vcur = d_cellVertices[ n_idx(cneigh-1,cellSet.w) ];
    for (int cn = 0; cn < cneigh; ++cn)
        {
        vnext = d_cellVertices[n_idx(cn,cellSet.w)];
        if(vcur == vertex2) break;
        vlast = vcur;
        vcur = vnext;
        };
    vertexSet.y=vnext;

    d_cellEdgeFlips[cells.x] = 0;
    d_cellEdgeFlips[cells.y] = 0;
    d_cellEdgeFlips[cells.z] = 0;
    d_cellEdgeFlips[cells.w] = 0;
    /*
if(vertex1 < 10)
{
printf("c123(%i,%i,%i)\n",cell1,cell2,cell3);
printf("cells and vertices: c(%i,%i,%i,%i) v(%i,%i,%i,%i)\n",cellSet.x,cellSet.y,cellSet.z,cellSet.w,vertex1,vertex2,vertexSet.x,vertexSet.y);
printf("icells %i %i %i\n",cell1, cell2, cell3);
printf("icells %i %i %i %i\n",cellSet.x, cellSet.y, cellSet.z,cellSet.w);
    printf("v1 = %i, v2 = %i\n",vertex1, vertex2);
    printf("vertex 1 v neighs: (%i,%i,%i)\n",d_vertexNeighbors[3*vertex1],d_vertexNeighbors[3*vertex1+1],d_vertexNeighbors[3*vertex1+2]);
    printf("vertex 2 v neighs: (%i,%i,%i)\n",d_vertexNeighbors[3*vertex2],d_vertexNeighbors[3*vertex2+1],d_vertexNeighbors[3*vertex2+2]);

cneigh = d_cellVertexNum[cell1];
printf("cell 1 (%i - %i): ",cell1,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cell1)]);
cneigh = d_cellVertexNum[cellSet.x];
printf("\ncell i (%i - %i): ",cellSet.x,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.x)]);
cneigh = d_cellVertexNum[cellSet.y];
printf("\ncell j (%i - %i): ",cellSet.y,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.y)]);
cneigh = d_cellVertexNum[cellSet.z];
printf("\ncell k (%i - %i): ",cellSet.z,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.z)]);
cneigh = d_cellVertexNum[cellSet.w];
printf("\ncell l (%i - %i): ",cellSet.w,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.w)]);
};
*/
    //forbid a T1 transition that would shrink a triangular cell
    if (d_cellVertexNum[cellSet.x] ==3 || d_cellVertexNum[cellSet.z] ==3)
        return;
    if(cellSet.x <0 || cellSet.y < 0 || cellSet.z <0 || cellSet.w <0)
        return;

    //okay, we're ready to go. First, rotate the vertices in the edge and set them at twice their original distance
    Dscalar2 edge;
    Dscalar2 v1 = d_vertexPositions[vertex1];
    Dscalar2 v2 = d_vertexPositions[vertex2];
    Box.minDist(v1,v2,edge);

    Dscalar2 midpoint;
    midpoint.x = v2.x + 0.5*edge.x;
    midpoint.y = v2.y + 0.5*edge.y;

    v1.x = midpoint.x-edge.y;v1.y = midpoint.y+edge.x;
    v2.x = midpoint.x+edge.y;v2.y = midpoint.y-edge.x;
    Box.putInBoxReal(v1);
    Box.putInBoxReal(v2);
    d_vertexPositions[vertex1] = v1;
    d_vertexPositions[vertex2] = v2;

//if(v1.x > 10) printf("%i %i, (%i,%i)\n",vertex1,vertex2,vertexSet.x,vertexSet.y);

    //now, re-wire the cells and vertices
    //start with the vertex-vertex and vertex-cell  neighbors
    for (int vert = 0; vert < 3; ++vert)
        {
        //vertex-cell neighbors
        if(d_vertexCellNeighbors[3*vertex1+vert] == cellSet.z)
            d_vertexCellNeighbors[3*vertex1+vert] = cellSet.w;
        if(d_vertexCellNeighbors[3*vertex2+vert] == cellSet.x)
            d_vertexCellNeighbors[3*vertex2+vert] = cellSet.y;

        //vertex-vertex neighbors
        if(d_vertexNeighbors[3*vertex1+vert] == vertexSet.x)
            d_vertexNeighbors[3*vertex1+vert] = vertexSet.y;
        if(d_vertexNeighbors[3*vertex2+vert] == vertexSet.y)
            d_vertexNeighbors[3*vertex2+vert] = vertexSet.x;

        if(d_vertexNeighbors[3*vertexSet.x+vert] == vertex1)
            d_vertexNeighbors[3*vertexSet.x+vert] = vertex2;
        if(d_vertexNeighbors[3*vertexSet.y+vert] == vertex2)
            d_vertexNeighbors[3*vertexSet.y+vert] = vertex1;
        };
    //now rewire the cells...
    //cell i loses v2 as a neighbor
    cneigh = d_cellVertexNum[cellSet.x];
    int cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cellVertices[n_idx(cc,cellSet.x)] == vertex2)
            cidx +=1;
        d_cellVertices[n_idx(cc,cellSet.x)] = d_cellVertices[n_idx(cidx,cellSet.x)];
        cidx +=1;
        };
    d_cellVertexNum[cellSet.x] -= 1;

    //cell j gains v2 in between v1 and b, so step through list backwards and insert
    cneigh = d_cellVertexNum[cellSet.y];
    bool found0 = false;
    for (int cc = cneigh-1; cc >= 0; --cc)
        {
        int cellIndex = d_cellVertices[n_idx(cc,cellSet.y)];
        if(!found0)
            d_cellVertices[n_idx(cc+1,cellSet.y)] = cellIndex;
        if(cellIndex == vertexSet.x)
            {
            found0 = true;
            d_cellVertices[n_idx(cc,cellSet.y)] = vertex2;
            }
        }
    d_cellVertexNum[cellSet.y] += 1;

    //cell k loses v1 as a neighbor
    cneigh = d_cellVertexNum[cellSet.z];
    cidx = 0;
    for (int cc = 0; cc < cneigh-1; ++cc)
        {
        if(d_cellVertices[n_idx(cc,cellSet.z)] == vertex1)
            cidx +=1;
        d_cellVertices[n_idx(cc,cellSet.z)] = d_cellVertices[n_idx(cidx,cellSet.z)];
        cidx +=1;
        };
    d_cellVertexNum[cellSet.z] -= 1;

    //cell l gains v1 in between v2 and a...copy the logic of cell j
    cneigh = d_cellVertexNum[cellSet.w];
    bool found = false;
    for (int cc = cneigh-1; cc >= 0; --cc)
        {
        int cellIndex = d_cellVertices[n_idx(cc,cellSet.w)];
        if(!found)
            d_cellVertices[n_idx(cc+1,cellSet.w)] = cellIndex;
        if(cellIndex == vertexSet.y)
            {
            found = true;
            d_cellVertices[n_idx(cc,cellSet.w)] = vertex1;
            }
        }
    d_cellVertexNum[cellSet.w] += 1;

/*
if(vertex1 < 10)
{
    printf("\n post swap \n");
    printf("v1 = %i, v2 = %i\n",vertex1, vertex2);
    printf("vertex 1 v neighs: (%i,%i,%i)\n",d_vertexNeighbors[3*vertex1],d_vertexNeighbors[3*vertex1+1],d_vertexNeighbors[3*vertex1+2]);
    printf("vertex 2 v neighs: (%i,%i,%i)\n",d_vertexNeighbors[3*vertex2],d_vertexNeighbors[3*vertex2+1],d_vertexNeighbors[3*vertex2+2]);

cneigh = d_cellVertexNum[cellSet.x];
printf("cell i (%i - %i): ",cellSet.x,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.x)]);
cneigh = d_cellVertexNum[cellSet.y];
printf("\ncell j (%i - %i): ",cellSet.y,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.y)]);
cneigh = d_cellVertexNum[cellSet.z];
printf("\ncell k (%i - %i): ",cellSet.z,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.z)]);
cneigh = d_cellVertexNum[cellSet.w];
printf("\ncell l (%i - %i): ",cellSet.w,cneigh);
for (int cn = 0; cn < cneigh; ++cn) printf("%i\t",d_cellVertices[n_idx(cn,cellSet.w)]);
};
*/
    };

//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_vm_geometry(
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_vertexCellNeighbors,
                    Dscalar2 *d_voroCur,
                    Dscalar4 *d_voroLastNext,
                    Dscalar2 *d_AreaPerimeter,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    vm_geometry_kernel<<<nblocks,block_size>>>(d_vertexPositions,
                                               d_cellVertexNum,d_cellVertices,
                                               d_vertexCellNeighbors,d_voroCur,
                                               d_voroLastNext,d_AreaPerimeter,
                                               N, n_idx, Box);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to test every edge for a T1 event, see if vertexMax needs to increase
bool gpu_vm_test_edges_for_T1(
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    gpubox   &Box,
                    Dscalar  T1THRESHOLD,
                    int      Nvertices,
                    int      vertexMax,
                    int      *d_grow,
                    Index2D  &n_idx)
    {
    unsigned int block_size = 128;
    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    unsigned int nblocks  = NvTimes3/block_size + 1;

    //test edges
    vm_simple_T1_test_kernel<<<nblocks,block_size>>>(
                                                      d_vertexPositions,d_vertexNeighbors,
                                                      d_vertexEdgeFlips,d_vertexCellNeighbors,
                                                      d_cellVertexNum,
                                                      Box,T1THRESHOLD,
                                                      NvTimes3,vertexMax,d_grow);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


//!determine whether any edges need to be flipped, and if we need to loop through the flipping routine, writing to d_finishedFlippingEdges the current state
bool gpu_vm_parse_multiple_flips(
                    int      *d_vertexEdgeFlips,
                    int      *d_vertexEdgeFlipsCurrent,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_finishedFlippingEdges,
                    int      *d_cellEdgeFlips,
                    int4     *d_cellSets,
                    Index2D  &n_idx,
                    int      Ncells)
    {
    unsigned int block_size = 128;

    /*The issue is that if a cell is involved in two edge flips done by different threads, the resulting
    data structure for what vertices belong to cells and what cells border which vertex will be
    inconsistently updated.

    The strategy will be to take the d_vertexEdgeFlips list, put at most one T1 per cell per vertex into the
    d_vertexEdgeFlipsCurrent list (erasing it from the d_vertexEdgeFlips list), and swap the edges specified
    by the "current" list. If d_vertexEdgeFlips is empty, we will set d_finishedFlippingEdges[0] to 1,
     and if any cell has multiple edges to flip, we set d_finishedFlippingEdges[1] to 1. As long
    as the zeroth entry is 1, the flip edges kernel is called; as long as the first entry is 1 the cpp code will continue calling this gpu_avm_flip_edges function.
    */

    //first select a few edges to flip...
    if(Ncells <128) block_size = 32;
    unsigned int nblocks = Ncells/block_size + 1;
    vm_one_T1_per_cell_per_vertex_kernel<<<nblocks,block_size>>>(
                                                                d_vertexEdgeFlips,
                                                                d_vertexEdgeFlipsCurrent,
                                                                d_vertexNeighbors,
                                                                d_vertexCellNeighbors,
                                                                d_cellVertexNum,
                                                                d_cellVertices,
                                                                d_finishedFlippingEdges,
                                                                d_cellEdgeFlips,
                                                                d_cellSets,
                                                                n_idx,
                                                                Ncells);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


//!Call the kernel to flip at most one edge per cell, write to d_finishedFlippingEdges the current state
bool gpu_vm_flip_edges(
                    int      *d_vertexEdgeFlipsCurrent,
                    Dscalar2 *d_vertexPositions,
                    int      *d_vertexNeighbors,
                    int      *d_vertexCellNeighbors,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      *d_cellEdgeFlips,
                    int4     *d_cellSets,
                    gpubox   &Box,
                    Index2D  &n_idx,
                    int      Nvertices,
                    int      Ncells)
    {
    unsigned int block_size = 128;

    if(Ncells <128) block_size = 32;

    int NvTimes3 = Nvertices*3;
    if (NvTimes3 < 128) block_size = 32;
    unsigned int nblocks  = NvTimes3/block_size + 1;

    vm_flip_edges_kernel<<<nblocks,block_size>>>(
                                                  d_vertexEdgeFlipsCurrent,d_vertexPositions,d_vertexNeighbors,
                                                  d_vertexCellNeighbors,d_cellVertexNum,d_cellVertices,d_cellEdgeFlips,d_cellSets,
                                                  Box,
                                                  n_idx,NvTimes3);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to calculate the position of each cell from the position of its vertices
bool gpu_vm_get_cell_positions(
                    Dscalar2 *d_cellPositions,
                    Dscalar2 *d_vertexPositions,
                    int      *d_cellVertexNum,
                    int      *d_cellVertices,
                    int      N,
                    Index2D  &n_idx,
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    vm_get_cell_positions_kernel<<<nblocks,block_size>>>(d_cellPositions,d_vertexPositions,
                                                          d_cellVertexNum,d_cellVertices,
                                                          N, n_idx, Box);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
