#ifndef __vertexQuadraticEnergyWithTension_CUH__
#define __vertexQuadraticEnergyWithTension_CUH__

#include "functions.h"
#include "indexer.h"

/*!
 \file vertexQuadraticEnergyWithTension.cuh
A file providing an interface to the relevant cuda calls for the VertexQuadraticEnergy class
*/

/** @defgroup vmKernels vertex model Kernels
 * @{
 * \brief CUDA kernels and callers for 2D vertex models
 */

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
            double KA, double KP);

#endif
