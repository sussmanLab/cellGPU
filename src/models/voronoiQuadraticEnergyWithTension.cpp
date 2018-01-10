#define ENABLE_CUDA

#include "voronoiQuadraticEnergyWithTension.h"
#include "voronoiQuadraticEnergyWithTension.cuh"
/*! \file voronoiQuadraticEnergyWithTension.cpp */


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
void VoronoiQuadraticEnergyWithTension::setSurfaceTension(vector<Dscalar> gammas)
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
void VoronoiQuadraticEnergyWithTension::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    computeGeometry();
    if (GPUcompute)
        {
        ComputeForceSetsGPU();
        SumForcesGPU();
        }
    else
        {
        if(Tension)
            {
            if (simpleTension)
                {
                for (int ii = 0; ii < Ncells; ++ii)
                    computeVoronoiSimpleTensionForceCPU(ii);
                }
            else
                {
                for (int ii = 0; ii < Ncells; ++ii)
                    computeVoronoiTensionForceCPU(ii);
                };
            }
        else
            {
            for (int ii = 0; ii < Ncells; ++ii)
                computeVoronoiForceCPU(ii);
            };
        };
    };


/*!
\pre The geoemtry (area and perimeter) has already been calculated
\post calculate the contribution to the net force on every particle from each of its voronoi vertices
via a cuda call
*/
void VoronoiQuadraticEnergyWithTension::ComputeForceSetsGPU()
    {
        if(Tension)
            {
            if (simpleTension)
                computeVoronoiSimpleTensionForceSetsGPU();
            else
                computeVoronoiTensionForceSetsGPU();
            }
        else
            computeVoronoiForceSetsGPU();
    };

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2 + \sum_{[i]\neq[j]} \gamma_{[i][j]}l_{ij}
*/
Dscalar VoronoiQuadraticEnergyWithTension::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    //first, compute the area and perimeter pieces...which are easy
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    Energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        Energy += KA * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        Energy += KP * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };

    //now, the potential line tension terms
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);
    for (int cell = 0; cell < Ncells; ++cell)
        {
        //get the Delaunay neighbors of the cell
        int neigh = h_nn.data[cell];
        vector<int> ns(neigh);
        vector<Dscalar2> voro(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn] = h_n.data[n_idx(nn,cell)];
            voro[nn] = h_v.data[n_idx(nn,cell)];
            };

        Dscalar2 vlast, vnext,vcur;
        Dscalar2 dlast, dnext;
        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vcur = voro[nn];
            vnext = voro[(nn+1)%neigh];
            int baseNeigh = ns[nn];
            int typeI = h_ct.data[cell];
            int typeK = h_ct.data[baseNeigh];
            //if the cell types are different, calculate everything once
            if (typeI != typeK && cell < baseNeigh)
                {
                dnext.x = vcur.x-vnext.x;
                dnext.y = vcur.y-vnext.y;
                Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
                if (simpleTension)
                    Energy += dnnorm*gamma;
                else
                    Energy += dnnorm*h_tm.data[cellTypeIndexer(typeK,typeI)];
                };
            vlast=vcur;
            };
        };
    return Energy;
    };


/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices
*/
void VoronoiQuadraticEnergyWithTension::computeVoronoiSimpleTensionForceSetsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);

    gpu_VoronoiSimpleTension_force_sets(
                    d_p.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_vc.data,
                    d_vln.data,
                    d_forceSets.data,
                    d_nidx.data,
                    d_ct.data,
                    KA,
                    KP,
                    gamma,
                    NeighIdxNum,n_idx,*(Box));
    };

/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices, using the general surface tension matrix
*/
void VoronoiQuadraticEnergyWithTension::computeVoronoiTensionForceSetsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);

    ArrayHandle<Dscalar> d_tm(tensionMatrix,access_location::device,access_mode::read);

    gpu_VoronoiTension_force_sets(
                    d_p.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_vc.data,
                    d_vln.data,
                    d_forceSets.data,
                    d_nidx.data,
                    d_ct.data,
                    d_tm.data,
                    cellTypeIndexer,
                    KA,
                    KP,
                    NeighIdxNum,n_idx,*(Box));
    };
/*!
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void VoronoiQuadraticEnergyWithTension::computeVoronoiSimpleTensionForceCPU(int i)
    {
    Dscalar Pthreshold = THRESHOLD;
    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
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
    Box->minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        int id = n_idx(nn,i);
        nnextp = h_p.data[ns[nn]];
        Box->minDist(nnextp,pi,rik);
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


        Dscalar2 dAidv,dPidv,dTidv;
        dTidv.x = 0.0;
        dTidv.y = 0.0;
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

        //individual line tensions
        if(h_ct.data[i] != h_ct.data[baseNeigh])
            {
            dTidv.x -= dnext.x/dnnorm;
            dTidv.y -= dnext.y/dnnorm;
            };
        if(h_ct.data[i] != h_ct.data[otherNeigh])
            {
            dTidv.x += dlast.x/dlnorm;
            dTidv.y += dlast.y/dlnorm;
            };
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
        Box->minDist(nl1,pi,r1);
        Box->minDist(nn1,pi,r2);
        Box->minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        Dscalar Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        Dscalar Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        Dscalar Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        Dscalar Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        Dscalar2 dAkdv,dPkdv,dTkdv;
        dTkdv.x = 0.0;
        dTkdv.y = 0.0;
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

        if(h_ct.data[i]!=h_ct.data[baseNeigh])
            {
            dTkdv.x +=dlast.x/dlnorm;
            dTkdv.y +=dlast.y/dlnorm;
            };
        if(h_ct.data[otherNeigh]!=h_ct.data[baseNeigh])
            {
            dTkdv.x -=dnext.x/dnnorm;
            dTkdv.y -=dnext.y/dnnorm;
            };
            
        Dscalar2 dAjdv,dPjdv,dTjdv;
        dTjdv.x = 0.0;
        dTjdv.y = 0.0;
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

        if(h_ct.data[i]!=h_ct.data[otherNeigh])
            {
            dTjdv.x -=dnext.x/dnnorm;
            dTjdv.y -=dnext.y/dnnorm;
            };
        if(h_ct.data[otherNeigh]!=h_ct.data[baseNeigh])
            {
            dTjdv.x +=dlast.x/dlnorm;
            dTjdv.y +=dlast.y/dlnorm;
            };

        Dscalar2 dEdv;

        dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x + gamma*dTidv.x;
        dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y + gamma*dTidv.y;
        dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x + gamma*dTkdv.x;
        dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y + gamma*dTkdv.y;
        dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x + gamma*dTjdv.x;
        dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y + gamma*dTjdv.y;

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
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void VoronoiQuadraticEnergyWithTension::computeVoronoiTensionForceCPU(int i)
    {
    Dscalar Pthreshold = THRESHOLD;
    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_external_forces(external_forces,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_exes(exclusions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);

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
    Box->minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        int id = n_idx(nn,i);
        nnextp = h_p.data[ns[nn]];
        Box->minDist(nnextp,pi,rik);
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


        Dscalar2 dAidv,dPidv,dTidv;
        dTidv.x = 0.0;
        dTidv.y = 0.0;
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

        //individual line tensions
        int typeI = h_ct.data[i];
        int typeJ = h_ct.data[otherNeigh];
        int typeK = h_ct.data[baseNeigh];
        if(typeI != typeK)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeK,typeI)];
            dTidv.x -= g*dnext.x/dnnorm;
            dTidv.y -= g*dnext.y/dnnorm;
            };
        if(typeI != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeI)];
            dTidv.x += g*dlast.x/dlnorm;
            dTidv.y += g*dlast.y/dlnorm;
            };
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
        Box->minDist(nl1,pi,r1);
        Box->minDist(nn1,pi,r2);
        Box->minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        Dscalar Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        Dscalar Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        Dscalar Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        Dscalar Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        Dscalar2 dAkdv,dPkdv,dTkdv;
        dTkdv.x = 0.0;
        dTkdv.y = 0.0;
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

        if(typeI != typeK)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeK,typeI)];
            dTkdv.x += g*dlast.x/dlnorm;
            dTkdv.y += g*dlast.y/dlnorm;
            };
        if(typeK != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeK)];
            dTkdv.x -= g*dnext.x/dnnorm;
            dTkdv.y -= g*dnext.y/dnnorm;
            };
            
        Dscalar2 dAjdv,dPjdv,dTjdv;
        dTjdv.x = 0.0;
        dTjdv.y = 0.0;
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

        if(typeI != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeI)];
            dTjdv.x -= g*dnext.x/dnnorm;
            dTjdv.y -= g*dnext.y/dnnorm;
            };
        if(typeK != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeK)];
            dTjdv.x += g*dlast.x/dlnorm;
            dTjdv.y += g*dlast.y/dlnorm;
            };

        Dscalar2 dEdv;

        dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x + dTidv.x;
        dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y + dTidv.y;
        dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x + dTkdv.x;
        dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y + dTkdv.y;
        dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x + dTjdv.x;
        dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y + dTjdv.y;

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
