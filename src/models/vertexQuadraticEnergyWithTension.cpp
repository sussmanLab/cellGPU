#define ENABLE_CUDA

#include "vertexQuadraticEnergyWithTension.h"
#include "vertexQuadraticEnergyWithTension.cuh"
/*! \file vertexQuadraticEnergyWithTension.cpp */

/*!
This function defines a matrix, \gamma_{i,j}, describing the imposed tension between cell types i and
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

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VertexQuadraticEnergyWithTension::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    computeGeometry();
    if (GPUcompute)
        {
        if (simpleTension)
            computeVertexSimpleTensionForceGPU();
        else if (Tension)
            computeVertexTensionForceGPU();
        }
    else
        {
        if(Tension)
                computeVertexTensionForcesCPU();
        else
            computeForcesCPU();
        };
    };

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex...for the cpu part combine the simple and complex tension routines
*/
void VertexQuadraticEnergyWithTension::computeVertexTensionForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its three cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        //for the change in the energy of the cell, just repeat the vertexQuadraticEnergy part
        int cellIdx1 = h_vcn.data[fsidx];
        Dscalar Adiff = KA*(h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x);
        Dscalar Pdiff = KP*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y);
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetVertexModel is defined in inc/utility/functions.h
        computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;

        //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
        int cellNeighs = h_cvn.data[cellIdx1];
        //find the index of vcur and vnext
        int vCurIdx = fsidx/3;
        int vNextInt = 0;
        if (h_cv.data[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
            {
            for (int nn = 0; nn < cellNeighs-1; ++nn)
                {
                int idx = h_cv.data[n_idx(nn,cellIdx1)];
                if (idx == vCurIdx)
                    vNextInt = nn +1;
                };
            };
        int vNextIdx = h_cv.data[n_idx(vNextInt,cellIdx1)];

        //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
        int cellIdx2 = 0;
        int cellOfSet = fsidx-3*vCurIdx;
        for (int cc = 0; cc < 3; ++cc)
            {
            if (cellOfSet == cc) continue;
            int cell2 = h_vcn.data[3*vCurIdx+cc];
            int cNeighs = h_cvn.data[cell2];
            for (int nn = 0; nn < cNeighs; ++nn)
                if (h_cv.data[n_idx(nn,cell2)] == vNextIdx)
                    cellIdx2 = cell2;
            }
        //now, determine the types of the two relevant cells, and add an extra force if needed
        int cellType1 = h_ct.data[cellIdx1];
        int cellType2 = h_ct.data[cellIdx2];
        if(cellType1 != cellType2)
            {
            Dscalar gammaEdge;
            if (simpleTension)
                gammaEdge = gamma;
            else
                gammaEdge = h_tm.data[cellTypeIndexer(cellType1,cellType2)];
            Dscalar2 dnext = vcur-vnext;
            Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            h_fs.data[fsidx].x -= gammaEdge*dnext.x/dnnorm;
            h_fs.data[fsidx].y -= gammaEdge*dnext.y/dnnorm;
            };
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };
    };

Dscalar VertexQuadraticEnergyWithTension::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    printf("computeEnergy function for VertexQuadraticEnergyWithTension not written. Very sorry\n");
    throw std::exception();
    return 0;
    };
void VertexQuadraticEnergyWithTension::computeVertexSimpleTensionForceGPU()
    {
    printf("computeVertexSimpleTensionForceGPU function not written. Very sorry\n");
    throw std::exception();
    };
void VertexQuadraticEnergyWithTension::computeVertexTensionForceGPU()
    {
    printf("computeVertexTensionForceGPU function not written. Very sorry\n");
    throw std::exception();
    };

