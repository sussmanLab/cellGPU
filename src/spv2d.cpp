#define ENABLE_CUDA

#include "spv2d.h"
#include "spv2d.cuh"
#include "cuda_profiler_api.h"
/*! \file spv2d.cpp */

/*!
\param n number of cells to initialize
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\param initGPURNG does the GPU RNG array need to be initialized?
\post Initialize(n,initGPURNcellsG) is called, as is setCellPreferenceUniform(1.0,4.0)
*/
SPV2D::SPV2D(int n, bool reprod,bool initGPURNG)
    {
    printf("Initializing %i cells with random positions in a square box...\n",n);
    Reproducible = reprod;
    Initialize(n,initGPURNG);
    setCellPreferencesUniform(1.0,4.0);
    };

/*!
\param n number of cells to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\param initGPURNG does the GPU RNG array need to be initialized?
\post Initialize(n,initGPURNG) is called
*/
SPV2D::SPV2D(int n,Dscalar A0, Dscalar P0,bool reprod,bool initGPURNG)
    {
    printf("Initializing %i cells with random positions in a square box... ",n);
    Reproducible = reprod;
    Initialize(n,initGPURNG);
    setCellPreferencesUniform(A0,P0);
    };

/*!
\param  n Number of cells to initialized
\param initGPU Should the GPU be initialized?
\post all GPUArrays are set to the correct size, v0 is set to 0.05, Dr is set to 1.0, the
Hilbert sorting period is set to -1 (i.e. off), the moduli are set to KA=KP=1.0, DelaunayMD is
initialized (initializeDelMD(n) gets called), particle exclusions are turned off, and auxiliary
data structures for the topology are set
*/
//take care of all class initialization functions
void SPV2D::Initialize(int n,bool initGPU)
    {
    Ncells=n;
    particleExclusions=false;
    Timestep = 0;
    triangletiming = 0.0; forcetiming = 0.0;
    setDeltaT(0.01);
    initializeDelMD(n);
    setModuliUniform(1.0,1.0);
    sortPeriod = -1;

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
    cellRNGs.resize(Ncells);
    if(initGPU)
        initializeCurandStates(Ncells,1337,Timestep);
    resetLists();
    allDelSets();

    //initialize the vectors passed to the e.o.m.s
    DscalarArrayInfo.push_back(cellDirectors);
    Dscalar2ArrayInfo.push_back(cellForces);
    Dscalar2ArrayInfo.push_back(Motility);
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
//    equationOfMotion->spatialSorting(itt);
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
Call all relevant functions to advance the system one time step; every sortPeriod also call the
spatial sorting routine.
\post The simulation is advanced one time step
*/
void SPV2D::performTimestep()
    {
    Timestep += 1;

    spatialSortThisStep = false;
    if (sortPeriod > 0)
        {
        if (Timestep % sortPeriod == 0)
            {
            spatialSortThisStep = true;
            };
        };

    computeForces();
    displaceCellsAndRotate();

    //spatial sorting also takes care of topology
    if (spatialSortThisStep)
        spatialSorting();
    else
        enforceTopology();
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
call the correct routines to move cells around and rotate the directors
*/
void SPV2D::displaceCellsAndRotate()
    {
    //swap in data for the equation of motion
    DscalarArrayInfo[0].swap(cellDirectors);
    Dscalar2ArrayInfo[0].swap(cellForces);
    Dscalar2ArrayInfo[1].swap(Motility);
    
    //call the equation of motion to get displacements
    equationOfMotion->integrateEquationsOfMotion(DscalarInfo,DscalarArrayInfo,Dscalar2ArrayInfo,displacements);
    //swap it back into the model
    DscalarArrayInfo[0].swap(cellDirectors);
    Dscalar2ArrayInfo[0].swap(cellForces);
    Dscalar2ArrayInfo[1].swap(Motility);

    //move the cells around
    moveDegreesOfFreedom(displacements);
/*
        calculateDispCPU();
*/
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
