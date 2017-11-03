#define ENABLE_CUDA

#include "vertexQuadraticEnergyWithTension.h"
#include "vertexQuadraticEnergyWithTension.cuh"
/*! \file vertexQuadraticEnergyWithTension.cpp */

/*!
This function definesa matrix, \gamma_{i,j}, describing the imposed tension between cell types i and
j. This function both sets that matrix and sets the flag telling the computeForces function to call
the more general tension force computations.
\pre the vector has n^2 elements, where n is the number of types, and the values of type in the system
must be of the form 0, 1, 2, ...n. The input vector should be laid out as:
gammas[0] = g_{0,0}  (an irrelevant value that is never called)
gammas[1] = g_{0,1}
gammas[n] = g_{0,n}
gammas[n+1] = g_{1,0} (physically, this better be the same as g_{0,1})
gammas[n+2] = g_{1,1} (again, never used)
...
gammas[n^2-1] = g_{n,n}
*/
void VertexQuadraticEnergyWithTension::setSurfaceTension(vector<Dscalar> gammas)
    {
    simpleTension = false;
    //set the tension matrix to the right size, and the indexer
    tensionMatrix.resize(gammas.size());
    int n = sqrt(gammas.size());
    cellTypeIndexer = Index2D(n);

    ArrayHandle<Dscalar> tensions(tensionMatrix,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < gammas.size(); ++ii)
        {   
        int typeI = ii/n;
        int typeJ = ii - typeI*n;
        tensions.data[cellTypeIndexer(typeJ,typeI)] = gammas[ii];
        };
    };

