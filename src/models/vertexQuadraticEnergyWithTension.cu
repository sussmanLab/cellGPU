#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "vertexQuadraticEnergyWithTension.cuh"

/** \file vertexQuadraticEnergyWithTension.cu
    * Defines kernel callers and kernels for GPU calculations of vertex model parts
*/

/*!
    \addtogroup vmKernels
    @{
*/

__global__ void vm_tensionForceSets_kernel(
            int *vertexCellNeighbors,
            double2 *voroCur,
            double4 *voroLastNext,
            double2 *areaPeri,
            double2 *APPref,
            int *cellType,
            int *cellVertices,
            int *cellVertexNum,
            double *tensionMatrix,
            double2 *forceSets,
            Index2D cellTypeIndexer,
            Index2D n_idx,
            bool simpleTension,
            double gamma,
            int nForceSets,
            double KA, double KP)
    
    {
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;
    
    //
    //first, compute the geometrical part of the force set using pre-computed data
    //
    double2 vlast,vcur,vnext,dEdv;

    int cellIdx1 = vertexCellNeighbors[fsidx];
    double Adiff = KA*(areaPeri[cellIdx1].x - APPref[cellIdx1].x);
    double Pdiff = KA*(areaPeri[cellIdx1].y - APPref[cellIdx1].y);
    vcur = voroCur[fsidx];
    vlast.x = voroLastNext[fsidx].x;  vlast.y = voroLastNext[fsidx].y;
    vnext.x = voroLastNext[fsidx].z;  vnext.y = voroLastNext[fsidx].w;

    computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
    forceSets[fsidx].x = dEdv.x;
    forceSets[fsidx].y = dEdv.y;

    //Now, to the potential for tension terms...
    //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
    int cellNeighs = cellVertexNum[cellIdx1];
    //find the index of vcur and vnext
    int vCurIdx = fsidx/3;
    int vNextInt = 0;
    if (cellVertices[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
        {
        for (int nn = 0; nn < cellNeighs-1; ++nn)
            {
            int idx = cellVertices[n_idx(nn,cellIdx1)];
            if (idx == vCurIdx)
                vNextInt = nn +1;
            };
        };
    int vNextIdx = cellVertices[n_idx(vNextInt,cellIdx1)];

    //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
    int cellIdx2 = 0;
    int cellOfSet = fsidx-3*vCurIdx;
    for (int cc = 0; cc < 3; ++cc)
        {
        if (cellOfSet == cc) continue;
        int cell2 = vertexCellNeighbors[3*vCurIdx+cc];
        int cNeighs = vertexCellNeighbors[cell2];
        for (int nn = 0; nn < cNeighs; ++nn)
            if (cellVertices[n_idx(nn,cell2)] == vNextIdx)
                cellIdx2 = cell2;
        }
    //now, determine the types of the two relevant cells, and add an extra force if needed
    int cellType1 = cellType[cellIdx1];
    int cellType2 = cellType[cellIdx2];
    if(cellType1 != cellType2)
        {
        double gammaEdge;
        if (simpleTension)
            gammaEdge = gamma;
        else
            gammaEdge = tensionMatrix[cellTypeIndexer(cellType1,cellType2)];
        double2 dnext = vcur-vnext;
        double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        forceSets[fsidx].x -= gammaEdge*dnext.x/dnnorm;
        forceSets[fsidx].y -= gammaEdge*dnext.y/dnnorm;
        };
    };

bool gpu_vertexModel_tension_force_sets(
        int *vertexCellNeighbors,
        double2 *voroCur,
        double4 *voroLastNext,
        double2 *areaPeri,
        double2 *APPref,
        int *cellType,
        int *cellVertices,
        int *cellVertexNum,
        double *tensionMatrix,
        double2 *forceSets,
        Index2D &cellTypeIndexer,
        Index2D &n_idx,
        bool simpleTension,
        double gamma,
        int nForceSets,
        double KA, double KP)
{
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks  = nForceSets/block_size + 1;

    vm_tensionForceSets_kernel<<<nblocks,block_size>>>(
            vertexCellNeighbors,voroCur,
            voroLastNext,areaPeri,APPref,
            cellType,cellVertices,cellVertexNum,
            tensionMatrix,forceSets,cellTypeIndexer,
            n_idx,simpleTension,gamma,
            nForceSets,KA,KP
            );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
};
/** @} */ //end of group declaration
