#define ENABLE_CUDA

#include "vertexGenericQuadraticEnergy.h"
/*! \file vertexGenericQuadraticEnergy.cpp */

/*!
\param n number of CELLS to initialize
*/
VertexGenericQuadraticEnergy::VertexGenericQuadraticEnergy(int n,bool reprod)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... \n",n);
    Reproducible = reprod;
    initializeVertexModelGenericBase(n);
    setCellPreferencesUniform(1.0,3.8);
    };

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2
*/
Dscalar VertexGenericQuadraticEnergy::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    Energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        Energy += KA * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        Energy += KP * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };
    return Energy;
    };

/*!
compute the geometry and the forces and the vertices, on either the GPU or CPU as determined by
flags
*/
void VertexGenericQuadraticEnergy::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    //compute the current area and perimeter of every cell
    computeGeometry();
    //use this information to compute the net force on the vertices
    if(GPUcompute)
        {
        computeForcesGPU();
        }
    else
        {
        computeForcesCPU();
        };
    };

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex
*/
void VertexGenericQuadraticEnergy::computeForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcnn(vertexCellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for (int vv = 0; vv < Nvertices; ++vv)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        int vcn = h_vcnn.data[vv];
        for (int cc = 0; cc < vcn; ++cc)
            {
            int fsidx = vertexNeighborIndexer(cc,vv);
            int cellIdx = h_vcn.data[fsidx];
            Dscalar Adiff = KA*(h_AP.data[cellIdx].x - h_APpref.data[cellIdx].x);
            Dscalar Pdiff = KP*(h_AP.data[cellIdx].y - h_APpref.data[cellIdx].y);
            vcur = h_vc.data[fsidx];
            vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
            vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;
            //computeForceSetVertexModel is defined in inc/utility/functions.h
            computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
            h_fs.data[fsidx].x = dEdv.x;
            h_fs.data[fsidx].y = dEdv.y;
            ftemp = ftemp + dEdv;
            };
        h_f.data[vv] = ftemp;
        };
    };

/*!
call kernels to (1) do force sets calculation, then (2) add them up
*/
void VertexGenericQuadraticEnergy::computeForcesGPU()
    {
    printf("computeForcesGPU in VertexGenericQuadraticEnergy function not written. Very sorry\n");
    throw std::exception();
    /*
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_fs(vertexForceSets,access_location::device, access_mode::overwrite);
    ArrayHandle<Dscalar2> d_f(vertexForces,access_location::device, access_mode::overwrite);

    int nForceSets = voroCur.getNumElements();
    gpu_avm_force_sets(
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    d_APpref.data,
                    d_fs.data,
                    nForceSets,
                    KA,
                    KP
                    );

    gpu_avm_sum_force_sets(
                    d_fs.data,
                    d_f.data,
                    Nvertices);
    */
    };
