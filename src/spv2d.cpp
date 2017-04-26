#define ENABLE_CUDA

#include "spv2d.h"
#include "spv2d.cuh"
#include "cuda_profiler_api.h"
/*! \file spv2d.cpp */

/*!
\param n number of cells to initialize
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post Initialize(n,initGPURNcellsG) is called, as is setCellPreferenceUniform(1.0,4.0)
*/
SPV2D::SPV2D(int n, bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box...\n",n);
    Reproducible = reprod;
    Initialize(n);
    setCellPreferencesUniform(1.0,4.0);
    };

/*!
\param n number of cells to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post Initialize(n,initGPURNG) is called
*/
SPV2D::SPV2D(int n,Dscalar A0, Dscalar P0,bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box... ",n);
    Reproducible = reprod;
    Initialize(n);
    setCellPreferencesUniform(A0,P0);
    };

/*!
\param  n Number of cells to initialized
\post all GPUArrays are set to the correct size, v0 is set to 0.05, Dr is set to 1.0, the
Hilbert sorting period is set to -1 (i.e. off), the moduli are set to KA=KP=1.0, DelaunayMD is
initialized (initializeDelMD(n) gets called), particle exclusions are turned off, and auxiliary
data structures for the topology are set
*/
//take care of all class initialization functions
void SPV2D::Initialize(int n)
    {
    Ncells=n;
    particleExclusions=false;
    Timestep = 0;
    triangletiming = 0.0; forcetiming = 0.0;
    setDeltaT(0.01);
    initializeDelMD(n);
    setModuliUniform(1.0,1.0);

    setv0Dr(0.05,1.0);
    cellForces.resize(n);
    external_forces.resize(n);
    AreaPeri.resize(n);
    CellType.resize(n);

    cellDirectors.resize(n);
    displacements.resize(n);

    vector<int> baseEx(n,0);
    setExclusions(baseEx);
    particleExclusions=false;

    setCellDirectorsRandomly();
    resetLists();
    allDelSets();
    };

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void SPV2D::computeForces()
    {
    if (GPUcompute)
        {
        computeGeometryGPU();
        ComputeForceSetsGPU();
        SumForcesGPU();
        }
    else
        {
        computeGeometryCPU();
        for (int ii = 0; ii < Ncells; ++ii)
            computeSPVForceCPU(ii);
        };
    };

/*!
goes through the process of testing and repairing the topology on either the CPU or GPU
\post and topological changes needed by cell motion are detected and repaired
*/
void SPV2D::enforceTopology()
    {
    if (GPUcompute)
        {
        testAndRepairTriangulation();
        ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::read);
        if(h_actf.data[0] == 1)
            {
            //maintain the auxilliary lists for computing forces
            if(completeRetriangulationPerformed || neighMaxChange)
                {
                if(neighMaxChange)
                    {
                    resetLists();
                    neighMaxChange = false;
                    };
                allDelSets();
                }
            else
                {
                bool localFail = false;
                for (int jj = 0;jj < NeedsFixing.size(); ++jj)
                    if(!getDelSets(NeedsFixing[jj]))
                        localFail=true;
                if (localFail)
                    {
                    cout << "Local triangulation failed to return a consistent set of topological information..." << endl;
                    cout << "Now attempting a global re-triangulation to save the day." << endl;
                    globalTriangulationCGAL();
                    //get new DelSets and DelOthers
                    resetLists();
                    allDelSets();
                    };
                };

            };
        //pre-copy some data back to device; this will overlap with some CPU time
        //...these are the arrays that are used by force_sets but not geometry, and should be switched to Async
        ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
        ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
        ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
        }
    else
        {
        testAndRepairTriangulation();
        if(neighMaxChange)
            {
            if(neighMaxChange)
                resetLists();
            neighMaxChange = false;
            allDelSets();
            };
        };
    };

/*!
When sortPeriod < 0, this routine does not get called
\post call Simple2DActiveCell's underlying Hilbert sort scheme, and re-index spv2d's extra arrays
*/
void SPV2D::spatialSorting()
    {
    spatiallySortCellsAndCellActivity();
    //reTriangulate with the new ordering
    globalTriangulationCGAL();
    //get new DelSets and DelOthers
    resetLists();
    allDelSets();

    //re-index all cell information arrays
    reIndexCellArray(exclusions);
    };

/*!
As the code is modified, all GPUArrays whose size depend on neighMax should be added to this function
\post voroCur,voroLastNext, delSets, delOther, and forceSets grow to size neighMax*Ncells
*/
void SPV2D::resetLists()
    {
    voroCur.resize(neighMax*Ncells);
    voroLastNext.resize(neighMax*Ncells);
    delSets.resize(neighMax*Ncells);
    delOther.resize(neighMax*Ncells);
    forceSets.resize(neighMax*Ncells);
    };

/*!
Calls updateNeighIdxs and then getDelSets(i) for all cells i
*/
void SPV2D::allDelSets()
    {
    updateNeighIdxs();
    for (int ii = 0; ii < Ncells; ++ii)
        getDelSets(ii);
    };

/*!
\param i the cell in question
\post the delSet and delOther data structure for cell i is updated. Recall that
delSet.data[n_idx(nn,i)] is an int2; the x and y parts store the index of the previous and next
Delaunay neighbor, ordered CCW. delOther contains the mutual neighbor of delSet.data[n_idx(nn,i)].y
and delSet.data[n_idx(nn,i)].z that isn't cell i
*/
bool SPV2D::getDelSets(int i)
    {
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int2> ds(delSets,access_location::host,access_mode::readwrite);
    ArrayHandle<int> dother(delOther,access_location::host,access_mode::readwrite);

    int iNeighs = neighnum.data[i];
    int nm2,nm1,n1,n2;
    nm2 = ns.data[n_idx(iNeighs-3,i)];
    nm1 = ns.data[n_idx(iNeighs-2,i)];
    n1 = ns.data[n_idx(iNeighs-1,i)];

    for (int nn = 0; nn < iNeighs; ++nn)
        {
        n2 = ns.data[n_idx(nn,i)];
        int nextNeighs = neighnum.data[n1];
        for (int nn2 = 0; nn2 < nextNeighs; ++nn2)
            {
            int testPoint = ns.data[n_idx(nn2,n1)];
            if(testPoint == nm1)
                {
                dother.data[n_idx(nn,i)] = ns.data[n_idx((nn2+1)%nextNeighs,n1)];
                break;
                };
            };
        ds.data[n_idx(nn,i)].x= nm1;
        ds.data[n_idx(nn,i)].y= n1;

        //is "delOther" a copy of i or either of the delSet points? if so, the local topology is inconsistent
        if(nm1 == dother.data[n_idx(nn,i)] || n1 == dother.data[n_idx(nn,i)] || i == dother.data[n_idx(nn,i)])
            return false;

        nm2=nm1;
        nm1=n1;
        n1=n2;

        };
    return true;
    };


/*!
\param exes a list of per-particle indications of whether a particle should be excluded (exes[i] !=0) or not/
*/
void SPV2D::setExclusions(vector<int> &exes)
    {
    particleExclusions=true;
    external_forces.resize(Ncells);
    exclusions.resize(Ncells);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ex(exclusions,access_location::host,access_mode::overwrite);

    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ex.data[ii] = 0;
        if( exes[ii] != 0)
            {
            //set v0 to zero and Dr to zero
            h_mot.data[ii].x = 0.0;
            h_mot.data[ii].y = 0.0;
            h_ex.data[ii] = 1;
            };
        };
    };

/*!
\pre The geoemtry (area and perimeter) has already been calculated
\post calculate the contribution to the net force on every particle from each of its voronoi vertices
via a cuda call
*/
void SPV2D::ComputeForceSetsGPU()
    {
        computeSPVForceSetsGPU();
    };

/*!
\pre forceSets are already computed
\post call the right routine to add up forceSets to get the net force per cell
*/
void SPV2D::SumForcesGPU()
    {
    if(!particleExclusions)
        sumForceSets();
    else
        sumForceSetsWithExclusions();
    };

/*!
\pre The topology of the Delaunay triangulation is up-to-date on the GPU
\post calculate all cell areas, perimenters, and voronoi neighbors
*/
void SPV2D::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_nn(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(cellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::overwrite);

    gpu_compute_geometry(
                        d_p.data,
                        d_AP.data,
                        d_nn.data,
                        d_n.data,
                        d_vc.data,
                        d_vln.data,
                        Ncells, n_idx,Box);
    };

/*!
\pre forceSets are already computed,
\post The forceSets are summed to get the net force per particle via a cuda call
*/
void SPV2D::sumForceSets()
    {

    ArrayHandle<int> d_nn(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forces(cellForces,access_location::device,access_mode::overwrite);

    gpu_sum_force_sets(
                    d_forceSets.data,
                    d_forces.data,
                    d_nn.data,
                    Ncells,n_idx);
    };

/*!
\pre forceSets are already computed, some particle exclusions have been defined.
\post The forceSets are summed to get the net force per particle via a cuda call, respecting exclusions
*/
void SPV2D::sumForceSetsWithExclusions()
    {

    ArrayHandle<int> d_nn(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forces(cellForces,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_external_forces(external_forces,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_exes(exclusions,access_location::device,access_mode::read);

    gpu_sum_force_sets_with_exclusions(
                    d_forceSets.data,
                    d_forces.data,
                    d_external_forces.data,
                    d_exes.data,
                    d_nn.data,
                    Ncells,n_idx);
    };

/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices
*/
void SPV2D::computeSPVForceSetsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);

    gpu_force_sets(
                    d_p.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_vc.data,
                    d_vln.data,
                    d_forceSets.data,
                    d_nidx.data,
                    KA,
                    KP,
                    NeighIdxNum,n_idx,Box);
    };

/*!
\pre Topology is up-to-date on the CPU
\post geometry and voronoi neighbor locations are computed for the current configuration
*/
void SPV2D::computeGeometryCPU()
    {
    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::overwrite);

    for (int i = 0; i < Ncells; ++i)
        {
        //get Delaunay neighbors of the cell
        int neigh = h_nn.data[i];
        vector<int> ns(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn]=h_n.data[n_idx(nn,i)];
            };

        //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
        vector<Dscalar2> voro(neigh);
        Dscalar2 circumcent;
        Dscalar2 nnextp,nlastp;
        Dscalar2 pi = h_p.data[i];
        Dscalar2 rij, rik;

        nlastp = h_p.data[ns[ns.size()-1]];
        Box.minDist(nlastp,pi,rij);
        for (int nn = 0; nn < neigh;++nn)
            {
            nnextp = h_p.data[ns[nn]];
            Box.minDist(nnextp,pi,rik);
            Circumcenter(rij,rik,circumcent);
            voro[nn] = circumcent;
            rij=rik;
            int id = n_idx(nn,i);
            h_v.data[id] = voro[nn];
            };

        Dscalar2 vlast,vnext;
        //compute Area and perimeter
        Dscalar Varea = 0.0;
        Dscalar Vperi = 0.0;
        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vnext=voro[nn];
            Varea += TriangleArea(vlast,vnext);
            Dscalar dx = vlast.x-vnext.x;
            Dscalar dy = vlast.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            vlast=vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

/*!
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void SPV2D::computeSPVForceCPU(int i)
    {
    Dscalar Pthreshold = THRESHOLD;

    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_external_forces(external_forces,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_exes(exclusions,access_location::host,access_mode::read);


    //get Delaunay neighbors of the cell
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
        };

    //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
    vector<Dscalar2> voro(neigh);
    vector<Matrix2x2> dhdri(neigh);
    Matrix2x2 Id;
    Dscalar2 circumcent;
    Dscalar2 rij,rik;
    Dscalar2 nnextp,nlastp;
    Dscalar2 rjk;
    Dscalar2 pi = h_p.data[i];

    nlastp = h_p.data[ns[ns.size()-1]];
    Box.minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        int id = n_idx(nn,i);
        nnextp = h_p.data[ns[nn]];
        Box.minDist(nnextp,pi,rik);
        voro[nn] = h_v.data[id];
        rjk.x =rik.x-rij.x;
        rjk.y =rik.y-rij.y;

        Dscalar2 dbDdri,dgDdri,dDdriOD,z;
        Dscalar betaD = -dot(rik,rik)*dot(rij,rjk);
        Dscalar gammaD = dot(rij,rij)*dot(rik,rjk);
        Dscalar cp = rij.x*rjk.y - rij.y*rjk.x;
        Dscalar D = 2*cp*cp;


        z.x = betaD*rij.x+gammaD*rik.x;
        z.y = betaD*rij.y+gammaD*rik.y;

        dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
        dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

        dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
        dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

        dDdriOD.x = (-2.0*rjk.y)/cp;
        dDdriOD.y = (2.0*rjk.x)/cp;

        dhdri[nn] = Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));

        rij=rik;
        };

    Dscalar2 vlast,vnext,vother;
    vlast = voro[neigh-1];

    //start calculating forces
    Dscalar2 forceSum;
    forceSum.x=0.0;forceSum.y=0.0;

    Dscalar Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
    Dscalar Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

    Dscalar2 vcur;
    vlast = voro[neigh-1];
    for(int nn = 0; nn < neigh; ++nn)
        {
        //first, let's do the self-term, dE_i/dr_i
        vcur = voro[nn];
        vnext = voro[(nn+1)%neigh];
        int baseNeigh = ns[nn];
        int other_idx = nn - 1;
        if (other_idx < 0) other_idx += neigh;
        int otherNeigh = ns[other_idx];


        Dscalar2 dAidv,dPidv;
        dAidv.x = 0.5*(vlast.y-vnext.y);
        dAidv.y = 0.5*(vnext.x-vlast.x);

        Dscalar2 dlast,dnext;
        dlast.x = vlast.x-vcur.x;
        dlast.y=vlast.y-vcur.y;

        Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

        dnext.x = vcur.x-vnext.x;
        dnext.y = vcur.y-vnext.y;
        Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;
        dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        //
        //now let's compute the other terms...first we need to find the third voronoi
        //position that v_cur is connected to
        //
        int neigh2 = h_nn.data[baseNeigh];
        int DT_other_idx=-1;
        for (int n2 = 0; n2 < neigh2; ++n2)
            {
            int testPoint = h_n.data[n_idx(n2,baseNeigh)];
            if(testPoint == otherNeigh) DT_other_idx = h_n.data[n_idx((n2+1)%neigh2,baseNeigh)];
            };
        if(DT_other_idx == otherNeigh || DT_other_idx == baseNeigh || DT_other_idx == -1)
            {
            printf("Triangulation problem %i\n",DT_other_idx);
            throw std::exception();
            };
        Dscalar2 nl1 = h_p.data[otherNeigh];
        Dscalar2 nn1 = h_p.data[baseNeigh];
        Dscalar2 no1 = h_p.data[DT_other_idx];

        Dscalar2 r1,r2,r3;
        Box.minDist(nl1,pi,r1);
        Box.minDist(nn1,pi,r2);
        Box.minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        Dscalar Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        Dscalar Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        Dscalar Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        Dscalar Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        Dscalar2 dAkdv,dPkdv;
        dAkdv.x = 0.5*(vnext.y-vother.y);
        dAkdv.y = 0.5*(vother.x-vnext.x);

        dlast.x = vnext.x-vcur.x;
        dlast.y=vnext.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vother.x;
        dnext.y = vcur.y-vother.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPkdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPkdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        Dscalar2 dAjdv,dPjdv;
        dAjdv.x = 0.5*(vother.y-vlast.y);
        dAjdv.y = 0.5*(vlast.x-vother.x);

        dlast.x = vother.x-vcur.x;
        dlast.y=vother.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vlast.x;
        dnext.y = vcur.y-vlast.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPjdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPjdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        Dscalar2 dEdv;

        dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x;
        dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y;
        dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x;
        dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y;
        dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x;
        dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y;

        Dscalar2 temp = dEdv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;

        vlast=vcur;
        };

    h_f.data[i].x=forceSum.x;
    h_f.data[i].y=forceSum.y;
    if(particleExclusions)
        {
        if(h_exes.data[i] != 0)
            {
            h_f.data[i].x = 0.0;
            h_f.data[i].y = 0.0;
            h_external_forces.data[i].x=-forceSum.x;
            h_external_forces.data[i].y=-forceSum.y;
            };
        }
    };

/*!
a utility function...output some information assuming the system is uniform
*/
void SPV2D::reportCellInfo()
    {
    printf("Ncells=%i\tv0=%f\tDr=%f\n",Ncells,v0,Dr);
    };

/*!
\param ri The position of cell i
\param rj The position of cell j
\param rk The position of cell k
\param jj the index EITHER 1 or 2 of the second derivative
Returns an 8-component vector containing the derivatives of the voronoi vertex formed by cells i, j, and k
with respect to r_i and r_{jj}... jj should be either 1 (to give d^2H/(d r_i)^2 or 2 (to give d^2H/dridrji)
The vector is laid out as
(H_x/r_{i,x}r_{j,x}, H_y/r_{i,x}r_{j,x}  
H_x/r_{i,y}r_{j,x}, H_y/r_{i,y}r_{j,x}  
H_x/r_{i,x}r_{j,y}, H_y/r_{i,x}r_{j,y}  
H_x/r_{i,y}r_{j,y}, H_y/r_{i,y}r_{j,y}  )
NOTE: This function does not check that ri, rj, and rk actually share a voronoi vertex in the triangulation
NOTE: This function assumes that rj and rk are w/r/t the position of ri, so ri = (0.,0.)
*/
vector<Dscalar> SPV2D::d2Hdridrj(Dscalar2 rj, Dscalar2 rk, int jj)
    {
    vector<Dscalar> answer(8);
    Dscalar hxr1xr2x, hyr1xr2x, hxr1yr2x,hyr1yr2x;
    Dscalar hxr1xr2y, hyr1xr2y, hxr1yr2y,hyr1yr2y;
    Dscalar rjx,rjy,rkx,rky;
    rjx = rj.x; rjy=rj.y; rkx=rk.x;rky=rk.y;

    Dscalar denominator;
    denominator = (rjx*rky-rjy*rkx)*(rjx*rky-rjy*rkx)*(rjx*rky-rjy*rkx);
    hxr1xr2x = hyr1xr2x = hxr1yr2x = hyr1yr2x = hxr1xr2y= hyr1xr2y= hxr1yr2y=hyr1yr2y= (1.0/denominator);
    //all derivatives in the dynMatTesting notebook
    //first, handle the d^2h/dr_i^2 case
    if ( jj == 1)
        {
        hxr1xr2x *= rjy*(rjy - rky)*rky*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky); 
        hyr1xr2x *= -(rjy*(rjx - rkx)*rky*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky));
        hxr1yr2x *= -((rjy - rky)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hyr1yr2x *= ((rjx - rkx)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hxr1xr2y *= -((rjy - rky)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) +rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hyr1xr2y *= ((rjx - rkx)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hxr1yr2y *= rjx*rkx*(rjy - rky)*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky);
        hyr1yr2y *= -(rjx*(rjx - rkx)*rkx*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky));
        }
    else
        {
        //next, handle the d^2h/dr_idr_j case
        hxr1xr2x *= rjy*(rjy - rky)*rky*(rjx*rkx + (rjy - rky)*rky - rkx*rkx);
        hyr1xr2x *= (-(rjy*rkx*rky*(-3*rjx*rkx + 3*(rjx*rjx) + rjy*rjy + 2*(rkx*rkx))) + rjy*rjy*(rkx*rkx*rkx) +(rjx*rjx*rjx - rjx*(rjy*rjy) + 3*rkx*(rjy*rjy))*(rky*rky) + rjy*(rjx - 2*rkx)*(rky*rky*rky))/2.;
        hxr1yr2x *= -((rjy - rky)*(rjy*rkx*((2*rjy - rky)*rky - rkx*rkx) + rjx*(2*rjy*(rkx*rkx) - rky*(rkx*rkx + rky*rky))))/2.;
        hyr1yr2x *= (rkx*(3*rjy*rkx*(rjx*rjx) - rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) +  rjx*(rky*(rjy*rjy) + rky*(rkx*rkx + rky*rky) - 2*rjy*(2*(rkx*rkx) + rky*rky))))/2.;
        hxr1xr2y *= -(rky*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-(rjy*rjy) + rkx*rkx + rky*rky) +rjx*(3*rky*(rjy*rjy) + rky*(rkx*rkx + rky*rky) - 2*rjy*(rkx*rkx + 2*(rky*rky)))))/2.;
        hyr1xr2y *= ((rjx - rkx)*(rjx*rky*(2*rjx*rkx - rkx*rkx - rky*rky) - rjy*(rkx*rkx*rkx - 2*rjx*(rky*rky) + rkx*(rky*rky))))/2.;
        hxr1yr2y *= (rkx*rky*(rjx*rjx*rjx) - rjy*rjy*rjy*(rkx*rkx) + rjx*rjx*(rjy*(rkx*rkx) - rky*(3*(rkx*rkx) + rky*rky)) +rjx*rkx*(3*rky*(rjy*rjy) + 2*rky*(rkx*rkx + rky*rky) - rjy*(rkx*rkx + 3*(rky*rky))))/2.;
        hyr1yr2y *= -(rjx*(rjx - rkx)*rkx*(rjx*rkx + (rjy - rky)*rky - rkx*rkx));
        };


    answer[0] = hxr1xr2x;
    answer[1] = hyr1xr2x;
    answer[2] = hxr1yr2x;
    answer[3] = hyr1yr2x;
    answer[4] = hxr1xr2y;
    answer[5] = hyr1xr2y;
    answer[6] = hxr1yr2y;
    answer[7] = hyr1yr2y;
    return answer;
    };

/*!
\param ri The position of cell i
\param rj The position of cell j
\param rk The position of cell k
Returns the derivative of the voronoi vertex shared by cells i, j , and k with respect to changing the position of cell i
the (row, column) format specifies dH_{row}/dr_{i,column}
*/
Matrix2x2 SPV2D::dHdri(Dscalar2 ri, Dscalar2 rj, Dscalar2 rk)
    {
    Matrix2x2 Id;
    Dscalar2 rij, rik, rjk;
    Box.minDist(rj,ri,rij);
    Box.minDist(rk,ri,rik);
    rjk.x =rik.x-rij.x;
    rjk.y =rik.y-rij.y;

    Dscalar2 dbDdri,dgDdri,dDdriOD,z;
    Dscalar betaD = -dot(rik,rik)*dot(rij,rjk);
    Dscalar gammaD = dot(rij,rij)*dot(rik,rjk);
    Dscalar cp = rij.x*rjk.y - rij.y*rjk.x;
    Dscalar D = 2*cp*cp;
    z.x = betaD*rij.x+gammaD*rik.x;
    z.y = betaD*rij.y+gammaD*rik.y;

    dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
    dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

    dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
    dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

    dDdriOD.x = (-2.0*rjk.y)/cp;
    dDdriOD.y = (2.0*rjk.x)/cp;

    return Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));
    };

/*!
\param i The index of cell i
\param j The index of cell j
\pre Requires that computeGeometry is current
Returns the derivative of the area of cell i w/r/t the position of cell j
*/
Dscalar2 SPV2D::dAidrj(int i, int j)
    {
    Dscalar2 answer;
    answer.x = 0.0; answer.y=0.0;
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::overwrite);

    //how many neighbors does cell i have?
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    bool jIsANeighbor = false;
    if (j ==i) jIsANeighbor = true;

    //which two vertices are important?
    int n1, n2;
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn] = h_n.data[n_idx(nn,i)];
        if (ns[nn] ==j)
            {
            jIsANeighbor = true;
            n1 = nn;
            n2 = nn+1;
            if (n2 ==neigh) n2 = 0;
            }
        };
    Dscalar2 vlast, vcur,vnext;
    //if j is not a neighbor of i (or i itself!) the  derivative vanishes
    if (!jIsANeighbor)
        return answer;
    //if i ==j, do the loop simply
    if ( i == j)
        {
        vlast = h_v.data[n_idx(neigh-1,i)];
        for (int vv = 0; vv < neigh; ++vv)
            {
            vcur = h_v.data[n_idx(vv,i)];
            vnext = h_v.data[n_idx((vv+1)%neigh,i)];
            Dscalar2 dAdv;
            dAdv.x = -0.5*(vlast.y-vnext.y);
            dAdv.y = -0.5*(vnext.x-vlast.x);

            int indexk = vv - 1;
            if (indexk <0) indexk = neigh-1;
            Dscalar2 temp = dAdv*dHdri(h_p.data[i],h_p.data[ h_n.data[n_idx(vv,i)] ],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x += temp.x;
            answer.y += temp.y;
            vlast = vcur;
            };
        return answer;
        };

    //otherwise, the interesting case
    vlast = h_v.data[n_idx(neigh-1,i)];
    for (int vv = 0; vv < neigh; ++vv)
        {
        vcur = h_v.data[n_idx(vv,i)];
        vnext = h_v.data[n_idx((vv+1)%neigh,i)];
        if(vv == n1 || vv == n2)
            {
            int indexk;
            if (vv == n1)
                indexk=vv-1;
            else
                indexk=vv;
                    
            if (indexk <0) indexk = neigh-1;
            Dscalar2 dAdv;
            dAdv.x = -0.5*(vlast.y-vnext.y);
            dAdv.y = -0.5*(vnext.x-vlast.x);
            Dscalar2 temp = dAdv*dHdri(h_p.data[j],h_p.data[i],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x += temp.x;
            answer.y += temp.y;
            };
        vlast = vcur;
        };
    return answer;
    }

/*!
\param i The index of cell i
\param j The index of cell j
Returns the derivative of the perimeter of cell i w/r/t the position of cell j
*/
Dscalar2 SPV2D::dPidrj(int i, int j)
    {
    Dscalar Pthreshold = THRESHOLD;
    Dscalar2 answer;
    answer.x = 0.0; answer.y=0.0;
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::overwrite);

    //how many neighbors does cell i have?
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    bool jIsANeighbor = false;
    if (j ==i) jIsANeighbor = true;

    //which two vertices are important?
    int n1, n2;
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn] = h_n.data[n_idx(nn,i)];
        if (ns[nn] ==j)
            {
            jIsANeighbor = true;
            n1 = nn;
            n2 = nn+1;
            if (n2 ==neigh) n2 = 0;
            }
        };
    Dscalar2 vlast, vcur,vnext;
    //if j is not a neighbor of i (or i itself!) the  derivative vanishes
    if (!jIsANeighbor)
        return answer;
    //if i ==j, do the loop simply
    if ( i == j)
        {
        vlast = h_v.data[n_idx(neigh-1,i)];
        for (int vv = 0; vv < neigh; ++vv)
            {
            vcur = h_v.data[n_idx(vv,i)];
            vnext = h_v.data[n_idx((vv+1)%neigh,i)];
            Dscalar2 dPdv;
            Dscalar2 dlast,dnext;
            dlast.x = vlast.x-vcur.x;
            dlast.y=vlast.y-vcur.y;

            Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

            dnext.x = vcur.x-vnext.x;
            dnext.y = vcur.y-vnext.y;
            Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            if(dnnorm < Pthreshold)
                dnnorm = Pthreshold;
            if(dlnorm < Pthreshold)
                dlnorm = Pthreshold;
            dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
            dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

            int indexk = vv - 1;
            if (indexk <0) indexk = neigh-1;
            Dscalar2 temp = dPdv*dHdri(h_p.data[i],h_p.data[ h_n.data[n_idx(vv,i)] ],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x -= temp.x;
            answer.y -= temp.y;
            vlast = vcur;
            };
        return answer;
        };

    //otherwise, the interesting case
    vlast = h_v.data[n_idx(neigh-1,i)];
    for (int vv = 0; vv < neigh; ++vv)
        {
        vcur = h_v.data[n_idx(vv,i)];
        vnext = h_v.data[n_idx((vv+1)%neigh,i)];
        if(vv == n1 || vv == n2)
            {
            int indexk;
            if (vv == n1)
                indexk=vv-1;
            else
                indexk=vv;
                    
            if (indexk <0) indexk = neigh-1;
            Dscalar2 dPdv;
            Dscalar2 dlast,dnext;
            dlast.x = vlast.x-vcur.x;
            dlast.y=vlast.y-vcur.y;

            Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

            dnext.x = vcur.x-vnext.x;
            dnext.y = vcur.y-vnext.y;
            Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            if(dnnorm < Pthreshold)
                dnnorm = Pthreshold;
            if(dlnorm < Pthreshold)
                dlnorm = Pthreshold;
            dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
            dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
            Dscalar2 temp = dPdv*dHdri(h_p.data[j],h_p.data[i],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x -= temp.x;
            answer.y -= temp.y;
            };
        vlast = vcur;
        };
    return answer;
    };

/*!
\param i The index of cell i
\param j The index of cell j
\pre Requires that computeGeometry is current
*/
Matrix2x2 SPV2D::d2Edridri(int i)
    {
    Matrix2x2  answer;
    answer.x11 = 0.0; answer.x12=0.0; answer.x21=0.0;answer.x22=0.0;

    return answer;
    };
/*!
\param i The index of cell i
\param j The index of cell j
\pre Requires that computeGeometry is current
The goal is to return a matrix (x11,x12,x21,x22) with
x11 = d^2 / dr_{i,x} dr_{j,x}
x12 = d^2 / dr_{i,x} dr_{j,y}
x21 = d^2 / dr_{i,y} dr_{j,x}
x22 = d^2 / dr_{i,y} dr_{j,y}
*/
Matrix2x2 SPV2D::d2Edridrj(int i, int j, neighborType neighbor)
    {
    Matrix2x2  answer;
    answer.x11 = 0.0; answer.x12=0.0; answer.x21=0.0;answer.x22=0.0;
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

    //how many neighbors does cell i have?
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn] = h_n.data[n_idx(nn,i)];
        };
    //the saved voronoi positions
    Dscalar2 vlast, vcur,vnext;
    //the cell indices
    int cellG, cellB,cellGp1,cellBm1;
    cellB = ns[neigh-1];
    cellBm1 = ns[neigh-2];
    vlast = h_v.data[n_idx(neigh-1,i)];
    Matrix2x2 tempMatrix;

    Dscalar dEdA = 2*KA*(h_AP.data[i].x - h_APpref.data[i].x);
    Dscalar2 dAadrj = dAidrj(i,j);
    answer = 2.0*KA*dyad(dAidrj(i,i),dAadrj);
    for (int vv = 0; vv < neigh; ++vv)
        {
        cellG = ns[vv];
        if (vv+1 == neigh)
            cellGp1 = ns[0];
        else
            cellGp1 = ns[vv+1];

        //first, what is the index and relative position of cell delta (which forms a vertex with gamma and beta connect by an edge to v_i)?
        int neigh2 = h_nn.data[cellG];
        int cellD=-1;
        for (int n2 = 0; n2 < neigh2; ++n2)
            {
            int testPoint = h_n.data[n_idx(n2,cellG)];
            if(testPoint == cellB) cellD = h_n.data[n_idx((n2+1)%neigh2,cellG)];
            };
        if(cellD == cellB || cellD  == cellG || cellD == -1)
            {
            printf("Triangulation problem %i\n",cellD);
            throw std::exception();
            };
        Dscalar2 rB,rG;
        Box.minDist(h_p.data[cellB],h_p.data[i],rB);
        Box.minDist(h_p.data[cellG],h_p.data[i],rG);
        Dscalar2 rD;
        Box.minDist(h_p.data[cellD],h_p.data[i],rD);
        Dscalar2 vother;
        Circumcenter(rB,rG,rD,vother);

        vcur = h_v.data[n_idx(vv,i)];
        vnext = h_v.data[n_idx((vv+1)%neigh,i)];

        Matrix2x2 dvidrj(0.0,0.0,0.0,0.0);
        Matrix2x2 dvidri = dHdri(h_p.data[i],h_p.data[cellB],h_p.data[cellG]);
        Matrix2x2 dvip1drj(0.0,0.0,0.0,0.0);
        Matrix2x2 dvim1drj(0.0,0.0,0.0,0.0);
        Matrix2x2 dvodrj(0.0,0.0,0.0,0.0);
        vector<Dscalar> d2vidridrj(8,0.0);
        if (neighbor == neighborType::self)
            {
            dvidrj = dHdri(h_p.data[i],h_p.data[cellB],h_p.data[cellG]);
            dvip1drj = dHdri(h_p.data[i],h_p.data[cellGp1],h_p.data[cellG]);
            dvim1drj = dHdri(h_p.data[i],h_p.data[cellBm1],h_p.data[cellB]);
            d2vidridrj = d2Hdridrj(rB,rG,1);
            };
        if (neighbor == neighborType::first)
            {
            if (j == cellG)
                {
                dvidrj = dHdri(h_p.data[cellG],h_p.data[i],h_p.data[cellB]);
                dvip1drj = dHdri(h_p.data[cellG],h_p.data[i],h_p.data[cellGp1]);
                dvodrj = dHdri(h_p.data[cellG],h_p.data[cellD],h_p.data[cellB]);
                //printf("%i\t%i\t%i\n %f\t%f\t%f\t%f\n\n",cellG,i,cellGp1,dvip1drj.x11,dvip1drj.x12,dvip1drj.x21,dvip1drj.x22);
                d2vidridrj = d2Hdridrj(rG,rB,2);
                };
            if (j == cellB)
                {
                dvidrj = dHdri(h_p.data[cellB],h_p.data[i],h_p.data[cellG]);
                dvim1drj = dHdri(h_p.data[cellB],h_p.data[i],h_p.data[cellBm1]);
                dvodrj = dHdri(h_p.data[cellB],h_p.data[cellD],h_p.data[cellG]);
                d2vidridrj = d2Hdridrj(rB,rG,2);
                };
            };
        /*
        for (int il = 0; il < 8; ++il)
            printf("%f, ",d2vidridrj[il]);
        int oo = cellB;
        if (j==cellB) oo=cellG;
        printf("\n %i %i %i\n", i, j, oo);
        */
        //
        //cell alpha terms
        //
        //Area part
        Dscalar2 dAdv;
        dAdv.x = -0.5*(vlast.y-vnext.y);
        dAdv.y = -0.5*(vnext.x-vlast.x);
        //first of three area terms... now done as a simple dyadic product outside the loop
        //answer += 2.*KA*dyad(dAdv*dvidri,dAadrj);

        //second of three area terms
        Matrix2x2 d2Advidrj;
        d2Advidrj.x11 = dvip1drj.x21-dvim1drj.x21;
        d2Advidrj.x12 = dvip1drj.x22-dvim1drj.x22;
        d2Advidrj.x21 = dvim1drj.x11-dvip1drj.x11;
        d2Advidrj.x22 = dvim1drj.x12-dvip1drj.x12;
        tempMatrix.x11 = d2Advidrj.x11*dvidri.x11+d2Advidrj.x21*dvidri.x21;
        tempMatrix.x12 = d2Advidrj.x12*dvidri.x11+d2Advidrj.x22*dvidri.x21;
        tempMatrix.x21 = d2Advidrj.x11*dvidri.x12+d2Advidrj.x21*dvidri.x22;
        tempMatrix.x22 = d2Advidrj.x12*dvidri.x12+d2Advidrj.x22*dvidri.x22;
        //printf("second terms: %f\t%f\t%f\t%f\n",tempMatrix.x11,tempMatrix.x12,tempMatrix.x21,tempMatrix.x22);
        answer += 0.5*dEdA*tempMatrix;


        //third of three area terms
        tempMatrix.x11 =dAdv.x*d2vidridrj[0]+dAdv.y*d2vidridrj[1]; 
        tempMatrix.x21 =dAdv.x*d2vidridrj[2]+dAdv.y*d2vidridrj[3]; 
        tempMatrix.x12 =dAdv.x*d2vidridrj[4]+dAdv.y*d2vidridrj[5]; 
        tempMatrix.x22 =dAdv.x*d2vidridrj[6]+dAdv.y*d2vidridrj[7]; 
        //printf("third terms: %f\t%f\t%f\t%f\n",tempMatrix.x11,tempMatrix.x12,tempMatrix.x21,tempMatrix.x22);
        answer += dEdA*tempMatrix;

        //perimeter part
        //first of three peri terms

        //second of three peri terms

        //third of three peri terms




        //now we compute terms related to cells gamma and beta
        //
        //cell gamma terms
        //
        //area part
        Dscalar dEGdA = 2.0*(h_AP.data[cellG].x  - h_APpref.data[cellG].x);
        Dscalar2 dAGdv;
        dAGdv.x = -0.5*(vnext.y-vother.y);
        dAGdv.y = -0.5*(vother.x-vnext.x);
        Dscalar2 dAGdrj = dAidrj(cellG,j);
        //first term
        answer += 2.*KA*dyad(dAGdv*dvidri,dAGdrj);
        //second term
        d2Advidrj.x11 = dvodrj.x21-dvip1drj.x21;
        d2Advidrj.x12 = dvodrj.x22-dvip1drj.x22;
        d2Advidrj.x21 = dvip1drj.x11-dvodrj.x11;
        d2Advidrj.x22 = dvip1drj.x12-dvodrj.x12;
        tempMatrix.x11 = d2Advidrj.x11*dvidri.x11+d2Advidrj.x21*dvidri.x21;
        tempMatrix.x12 = d2Advidrj.x12*dvidri.x11+d2Advidrj.x22*dvidri.x21;
        tempMatrix.x21 = d2Advidrj.x11*dvidri.x12+d2Advidrj.x21*dvidri.x22;
        tempMatrix.x22 = d2Advidrj.x12*dvidri.x12+d2Advidrj.x22*dvidri.x22;
        answer += 0.5*dEGdA*tempMatrix;

        //third term
        tempMatrix.x11 =dAGdv.x*d2vidridrj[0]+dAGdv.y*d2vidridrj[1]; 
        tempMatrix.x21 =dAGdv.x*d2vidridrj[2]+dAGdv.y*d2vidridrj[3]; 
        tempMatrix.x12 =dAGdv.x*d2vidridrj[4]+dAGdv.y*d2vidridrj[5]; 
        tempMatrix.x22 =dAGdv.x*d2vidridrj[6]+dAGdv.y*d2vidridrj[7]; 
        answer += dEGdA*tempMatrix;
        

        //perimeter part

        //
        //cell beta terms
        //
        //area terms
        Dscalar dEBdA = 2.0*(h_AP.data[cellB].x  - h_APpref.data[cellB].x);
        Dscalar2 dABdv;
        dABdv.x = 0.5*(vlast.y-vother.y);
        dABdv.y = 0.5*(vother.x-vlast.x);
        Dscalar2 dABdrj = dAidrj(cellB,j);
        
        //first term
        answer += 2.*KA*dyad(dABdv*dvidri,dABdrj);
        //second term
        d2Advidrj.x11 = dvim1drj.x21-dvodrj.x21;
        d2Advidrj.x12 = dvim1drj.x22-dvodrj.x22;
        d2Advidrj.x21 = dvodrj.x11-dvim1drj.x11;
        d2Advidrj.x22 = dvodrj.x12-dvim1drj.x12;
        tempMatrix.x11 = d2Advidrj.x11*dvidri.x11+d2Advidrj.x21*dvidri.x21;
        tempMatrix.x12 = d2Advidrj.x12*dvidri.x11+d2Advidrj.x22*dvidri.x21;
        tempMatrix.x21 = d2Advidrj.x11*dvidri.x12+d2Advidrj.x21*dvidri.x22;
        tempMatrix.x22 = d2Advidrj.x12*dvidri.x12+d2Advidrj.x22*dvidri.x22;
        answer += 0.5*dEBdA*tempMatrix;
        //third term
        tempMatrix.x11 =dABdv.x*d2vidridrj[0]+dABdv.y*d2vidridrj[1]; 
        tempMatrix.x21 =dABdv.x*d2vidridrj[2]+dABdv.y*d2vidridrj[3]; 
        tempMatrix.x12 =dABdv.x*d2vidridrj[4]+dABdv.y*d2vidridrj[5]; 
        tempMatrix.x22 =dABdv.x*d2vidridrj[6]+dABdv.y*d2vidridrj[7]; 
        answer += dEBdA*tempMatrix;

        //peri terms


        vlast=vcur;
        cellBm1=cellB;
        cellB=cellG;
        };

    return answer;
    };

