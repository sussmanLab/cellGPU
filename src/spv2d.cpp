using namespace std;

//definitions needed for DelaunayLoc, voroguppy namespace, Triangle, Triangle, and all GPU functions, respectively

#define EPSILON 1e-12
#define dbl float
#define REAL double
#define ANSI_DECLARATIONS
#define ENABLE_CUDA

#define PI 3.14159265358979323846

#include "spv2d.h"
#include "spv2d.cuh"
#include "cuda_profiler_api.h"

SPV2D::SPV2D(int n)
    {
    printf("Initializing %i cells with random positions in a square box\n",n);
    Initialize(n);
    };

SPV2D::SPV2D(int n,float A0, float P0)
    {
    printf("Initializing %i cells with random positions in a square box\n",n);
    Initialize(n);
    setCellPreferencesUniform(A0,P0);
    setCellTypeUniform(0);
    };

void SPV2D::Initialize(int n)
    {
    gamma = 0.;
    useTension = false;
    particleExclusions=false;
    Timestep = 0;
    triangletiming = 0.0; forcetiming = 0.0;
    setv0Dr(0.05,1.0);
    setDeltaT(0.01);
    initialize(n);
    forces.resize(n);
    external_forces.resize(n);
    AreaPeri.resize(n);

    cellDirectors.resize(n);
    cellDirectors_initial.resize(n);
    displacements.resize(n);


    ArrayHandle<float> h_cd(cellDirectors,access_location::host, access_mode::overwrite);
    ArrayHandle<float> h_cdi(cellDirectors_initial,access_location::host, access_mode::overwrite);

    int randmax = 100000000;
    for (int ii = 0; ii < N; ++ii)
        {
        float theta = 2.0*PI/(float)(randmax)* (float)(rand()%randmax);
        h_cd.data[ii] = theta;
        h_cdi.data[ii] = theta;
        };
    //setCurandStates(n);
    allDelSets();
    };

void SPV2D::allDelSets()
    {
    delSets.resize(neighMax*N);
    delOther.resize(neighMax*N);
    forceSets.resize(neighMax*N);
    for (int ii = 0; ii < N; ++ii)
        getDelSets(ii);
    };

void SPV2D::setCellPreferencesUniform(float A0, float P0)
    {
    AreaPeriPreferences.resize(N);
    ArrayHandle<float2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_p.data[ii].x = A0;
        h_p.data[ii].y = P0;
        };
    };

void SPV2D::setCellTypeUniform(int i)
    {
    CellType.resize(N);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_ct.data[ii] = i;
        };
    };

void SPV2D::setCellType(vector<int> &types)
    {
    CellType.resize(N);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_ct.data[ii] = types[ii];
        };
    };

void SPV2D::setCellTypeEllipse(float frac, float aspectRatio)
    {
    float x11,x12,x21,x22;
    Box.getBoxDims(x11,x12,x21,x22);
    float xc = x11*0.5;
    float yc = x22*0.5;

    float ry = sqrt(x11*x22*frac/(3.14159*aspectRatio));
    float rx = aspectRatio*ry;

    CellType.resize(N);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::overwrite);
    ArrayHandle<float2> h_p(points,access_location::host,access_mode::read);

    for (int ii = 0; ii < N; ++ii)
        {
        float px = h_p.data[ii].x;
        float py = h_p.data[ii].y;
        float test = (px-xc)*(px-xc)/(rx*rx) + (py-yc)*(py-yc)/(ry*ry);
        if (test <=1.0)
            h_ct.data[ii] = 0;
        else
            h_ct.data[ii] = 1;
        };
    };

void SPV2D::setv0Dr(float v0new,float drnew)
    {
    Motility.resize(N);
    ArrayHandle<float2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_mot.data[ii].x = v0new;
        h_mot.data[ii].y = drnew;
        };
    };
void SPV2D::setCellMotility(vector<float> &v0s,vector<float> &drs)
    {
    Motility.resize(N);
    ArrayHandle<float2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < N; ++ii)
        {
        h_mot.data[ii].x = v0s[ii];
        h_mot.data[ii].y = drs[ii];
        };
    };


void SPV2D::setExclusions(vector<int> &exes)
    {
    particleExclusions=true;
    external_forces.resize(N);
    exclusions.resize(N);
    ArrayHandle<float2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ex(exclusions,access_location::host,access_mode::overwrite);

    for (int ii = 0; ii < N; ++ii)
        {
        h_ex.data[ii] = 0;
        if( exes[ii] == 1)
            {
            //set v0 to zero and Dr to zero
            h_mot.data[ii].x = 0.0;
            h_mot.data[ii].y = 0.0;
            h_ex.data[ii] = 1;
            };
        };
    };
/*
void SPV2D::setCurandStates(int i)
    {
    gpu_init_curand(devStates,i,N);

    };
*/

/////////////////
//Utility
/////////////////

void SPV2D::getDelSets(int i)
    {
    ArrayHandle<int> neighnum(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighs,access_location::host,access_mode::read);
    ArrayHandle<int4> ds(delSets,access_location::host,access_mode::readwrite);
    ArrayHandle<int> dother(delOther,access_location::host,access_mode::readwrite);

    int iNeighs = neighnum.data[i];
    int nm2,nm1,n1,n2;
    if (iNeighs>3)

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
        ds.data[n_idx(nn,i)].x= nm2;
        ds.data[n_idx(nn,i)].y= nm1;
        ds.data[n_idx(nn,i)].z= n1;
        ds.data[n_idx(nn,i)].w= n2;

        nm2=nm1;
        nm1=n1;
        n1=n2;
        };
/*
    for (int nn = 0; nn < iNeighs; ++nn)
        {
        printf("%i  %i   %i   %i   ;  %i \n",ds.data[n_idx(nn,i)].x,ds.data[n_idx(nn,i)].y, ds.data[n_idx(nn,i)].z,ds.data[n_idx(nn,i)].w,dother.data[n_idx(nn,i)]);

        };
*/
    };


/////////////////
//Dynamics
/////////////////

void SPV2D::performTimestep()
    {
    Timestep += 1;
    if(GPUcompute)
        performTimestepGPU();
    else
        performTimestepCPU();
    };

void SPV2D::DisplacePointsAndRotate()
    {

    ArrayHandle<float2> d_p(points,access_location::device,access_mode::readwrite);
    ArrayHandle<float2> d_f(forces,access_location::device,access_mode::read);
    ArrayHandle<float> d_cd(cellDirectors,access_location::device,access_mode::readwrite);
    ArrayHandle<float2> d_motility(Motility,access_location::device,access_mode::read);

    gpu_displace_and_rotate(d_p.data,
                            d_f.data,
                            d_cd.data,
                            d_motility.data,
                            N,
                            deltaT,
                            Timestep,
//                            devStates,
                            Box);

    };

void SPV2D::centerCells()
    {
    ArrayHandle<float2> h_p(points,access_location::host,access_mode::read);
    float x11,x12,x21,x22;
    Box.getBoxDims(x11,x12,x21,x22);
    float xcm, ycm;
    xcm = 0.0; ycm = 0,0;
    for (int ii = 0; ii < N; ++ii)
        {
        float2 pos = h_p.data[ii];
        xcm+=pos.x;
        ycm+=pos.y;
        };
    xcm /= (float)N;
    ycm /= (float)N;
    if(true)
        {
        ArrayHandle<float2> h_disp(displacements,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < N; ++ii)
            {
            h_disp.data[ii].x = -(xcm-x11*0.5);
            h_disp.data[ii].y = -(ycm-x22*0.5);
            };
        };
    movePoints(displacements);
    };

void SPV2D::calculateDispCPU()
    {
    ArrayHandle<float2> h_f(forces,access_location::host,access_mode::read);
    ArrayHandle<float> h_cd(cellDirectors,access_location::host,access_mode::readwrite);
    ArrayHandle<float2> h_disp(displacements,access_location::host,access_mode::overwrite);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> normal(0.0,sqrt(2.0*deltaT*Dr));
    for (int ii = 0; ii < N; ++ii)
        {
        float dx,dy;
        float directorx = cos(h_cd.data[ii]);
        float directory = sin(h_cd.data[ii]);

        dx= deltaT*(v0*directorx+h_f.data[ii].x);
        dy= deltaT*(v0*directory+h_f.data[ii].y);
        h_disp.data[ii].x = dx;
        h_disp.data[ii].y = dy;

        //rotate each director a bit
        h_cd.data[ii] +=normal(gen);
        };

    //vector of displacements is forces*timestep + v0's*timestep


    };

void SPV2D::performTimestepCPU()
    {
//    printf("On CPU branch \n");
    computeGeometryCPU();
    if(useTension)
        {
        for (int ii = 0; ii < N; ++ii)
            computeSPVForceWithTensionsCPU(ii,gamma);
        }
    else
        {
        for (int ii = 0; ii < N; ++ii)
            computeSPVForceCPU(ii);
        };

    calculateDispCPU();

    movePoints(displacements);
    testAndRepairTriangulation(false);
    };

void SPV2D::performTimestepGPU()
    {
    clock_t t1,t2;
//    printf("computing geometry for timestep %i\n",Timestep);
    t1=clock();
    computeGeometry();
    t2=clock();
    triangletiming += (t2-t1);
//    printf("computing forces\n");
    t1=clock();
    cudaProfilerStart();
    if(useTension)
        computeSPVForceSetsWithTensionsGPU();
    else
        computeSPVForceSetsGPU();

    if(!particleExclusions)
        sumForceSets();
    else
        sumForceSetsWithExclusions();

    t2=clock();
    forcetiming += t2-t1;
    t1=clock();

//    printf("displacing particles\n");
    DisplacePointsAndRotate();


//    printf("recomputing triangulation\n");
    testAndRepairTriangulation();
    cudaProfilerStop();

    if(Fails == 1)
        {
        //maintain the auxilliary lists for computing forces
        if(delSets.getNumElements()!=neighMax*N)
            allDelSets();
        else
            {
            for (int jj = 0;jj < NeedsFixing.size(); ++jj)
                getDelSets(NeedsFixing[jj]);
            };
        };
    t2=clock();
    triangletiming += (t2-t1);

    };

void SPV2D::computeGeometry()
    {
    VoronoiPoints.resize(n_idx.getNumElements());
    ArrayHandle<float2> d_p(points,access_location::device,access_mode::read);
    ArrayHandle<float2> d_AP(AreaPeri,access_location::device,access_mode::readwrite);
    ArrayHandle<float2> d_voro(VoronoiPoints,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_nn(neigh_num,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(neighs,access_location::device,access_mode::read);

    gpu_compute_geometry(
                        d_p.data,
                        d_AP.data,
                        d_voro.data,
                        d_nn.data,
                        d_n.data,
                        N, n_idx,Box);


    };

void SPV2D::sumForceSets()
    {

    ArrayHandle<int> d_nn(neigh_num,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forceSets(forceSets,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forces(forces,access_location::device,access_mode::overwrite);

    gpu_sum_force_sets(
                    d_forceSets.data,
                    d_forces.data,
                    d_nn.data,
                    N,n_idx);
    };


void SPV2D::sumForceSetsWithExclusions()
    {

    ArrayHandle<int> d_nn(neigh_num,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forceSets(forceSets,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forces(forces,access_location::device,access_mode::overwrite);
    ArrayHandle<float2> d_external_forces(external_forces,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_exes(exclusions,access_location::device,access_mode::read);

    gpu_sum_force_sets_with_exclusions(
                    d_forceSets.data,
                    d_forces.data,
                    d_external_forces.data,
                    d_exes.data,
                    d_nn.data,
                    N,n_idx);
    };



void SPV2D::computeSPVForceSetsGPU()
    {
    ArrayHandle<float2> d_p(points,access_location::device,access_mode::read);
    ArrayHandle<int> d_nn(neigh_num,access_location::device,access_mode::read);
    ArrayHandle<float2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<float2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int4> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);


    float KA = 1.0;
    float KP = 1.0;
    gpu_force_sets(
                    d_p.data,
                    d_nn.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_forceSets.data,
                    KA,
                    KP,
                    N,neighMax,n_idx,Box);
    };


void SPV2D::computeSPVForceSetsWithTensionsGPU()
    {
    ArrayHandle<float2> d_p(points,access_location::device,access_mode::read);
    ArrayHandle<int> d_nn(neigh_num,access_location::device,access_mode::read);
    ArrayHandle<float2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<float2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int4> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<float2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_ct(CellType,access_location::device,access_mode::read);


    float KA = 1.0;
    float KP = 1.0;
    gpu_force_sets_tensions(
                    d_p.data,
                    d_nn.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_forceSets.data,
                    d_ct.data,
                    KA,
                    KP,
                    gamma,
                    N,neighMax,n_idx,Box);


    };


void SPV2D::computeGeometryCPU()
    {
    for (int i = 0; i < N; ++i)
        {
//        printf("cell %i:\n",i);
        //read in all the data we'll need
        ArrayHandle<float2> h_p(points,access_location::host,access_mode::read);
        ArrayHandle<float2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);

        ArrayHandle<int> h_nn(neigh_num,access_location::host,access_mode::read);
        ArrayHandle<int> h_n(neighs,access_location::host,access_mode::read);

        //get Delaunay neighbors of the cell
        int neigh = h_nn.data[i];
        vector<int> ns(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn]=h_n.data[n_idx(nn,i)];
            };

        //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
        vector<float2> voro(neigh);
        float2 circumcent;
        float2 origin; origin.x = 0.; origin.y=0.;
        float2 nnextp,nlastp;
        float2 pi = h_p.data[i];
        float2 rij, rik;

        nlastp = h_p.data[ns[ns.size()-1]];
        Box.minDist(nlastp,pi,rij);
        for (int nn = 0; nn < neigh;++nn)
            {
            nnextp = h_p.data[ns[nn]];
            Box.minDist(nnextp,pi,rik);
            Circumcenter(origin,rij,rik,circumcent);
            voro[nn] = circumcent;
            rij=rik;
            };

        float2 vlast,vnext;
        //compute Area and perimeter
        float Varea = 0.0;
        float Vperi = 0.0;
        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vnext=voro[nn];
            Varea += TriangleArea(vlast,vnext);
            float dx = vlast.x-vnext.x;
            float dy = vlast.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            vlast=vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

void SPV2D::computeSPVForceCPU(int i)
    {
    float Pthreshold = 1e-8;
 //   printf("cell %i: \n",i);
    //for testing these routines...
    vector <int> test;
    DelaunayCell celltest;
    delLoc.triangulatePoint(i, test,celltest);

    //read in all the data we'll need
    ArrayHandle<float2> h_p(points,access_location::host,access_mode::read);
    ArrayHandle<float2> h_f(forces,access_location::host,access_mode::readwrite);
    ArrayHandle<float2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<float2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighs,access_location::host,access_mode::read);

    //get Delaunay neighbors of the cell
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
   //     printf("%i - ",ns[nn]);
        };

    //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
    vector<float2> voro(neigh);
    vector<Matrix2x2> dhdri(neigh);
    Matrix2x2 Id;
    float2 circumcent;
    float2 origin; origin.x = 0.; origin.y=0.;
    float2 rij,rik;
    float2 nnextp,nlastp;
    float2 rjk;
    float2 pi = h_p.data[i];

    nlastp = h_p.data[ns[ns.size()-1]];
    Box.minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        nnextp = h_p.data[ns[nn]];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(origin,rij,rik,circumcent);
        voro[nn] = circumcent;
        rjk.x =rik.x-rij.x;
        rjk.y =rik.y-rij.y;

        float2 dbDdri,dgDdri,dDdriOD,z;
        float betaD = -dot(rik,rik)*dot(rij,rjk);
        float gammaD = dot(rij,rij)*dot(rik,rjk);
        float cp = rij.x*rjk.y - rij.y*rjk.x;
        float D = 2*cp*cp;


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

    float2 vlast,vnext,vother;
    vlast = voro[neigh-1];

    //start calculating forces
    float2 forceSum;
    forceSum.x=0.0;forceSum.y=0.0;
    float KA = 1.0;
    float KP = 1.0;

    float Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
    float Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

//    printf("cell %i: %f, %f\n",i,h_AP.data[i].x,h_APpref.data[i].x);

    float2 vcur;
    vlast = voro[neigh-1];
    for(int nn = 0; nn < neigh; ++nn)
        {
        //first, let's do the self-term, dE_i/dr_i
        vcur = voro[nn];
        vnext = voro[(nn+1)%neigh];
        float2 dEidv,dAidv,dPidv;
        dAidv.x = 0.5*(vlast.y-vnext.y);
        dAidv.y = 0.5*(vnext.x-vlast.x);

        float2 dlast,dnext;
        dlast.x = vlast.x-vcur.x;
        dlast.y=vlast.y-vcur.y;

        float dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

        dnext.x = vcur.x-vnext.x;
        dnext.y = vcur.y-vnext.y;
        float dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;



        //
        //
        //now let's compute the other terms...first we need to find the third voronoi
        //position that v_cur is connected to
        //
        //
        int baseNeigh = ns[nn];
        int other_idx = nn - 1;
        if (other_idx < 0) other_idx += neigh;
        int otherNeigh = ns[other_idx];
        int neigh2 = h_nn.data[baseNeigh];
        int DT_other_idx = -1;
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
        float2 nl1 = h_p.data[otherNeigh];
        float2 nn1 = h_p.data[baseNeigh];
        float2 no1 = h_p.data[DT_other_idx];


        float2 r1,r2,r3;
        Box.minDist(nl1,pi,r1);
        Box.minDist(nn1,pi,r2);
        Box.minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);
        if(vother.x*vother.x+vother.y*vother.y > 10)
//        if(true)
            {
//        printf("\nvother %i--%i--%i = (%f,%f)\n",baseNeigh,otherNeigh,DT_other_idx,vother.x,vother.y);
//            printf("Big voro_other:\n");
//            printf("(%f,%f), (%f,%f), (%f,%f)\n",r1.x,r1.y,r2.x,r2.y,r3.x,r3.y);

            };

        float Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        float Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        float Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        float Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        float2 dEkdv,dAkdv,dPkdv;
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

        float2 dEjdv,dAjdv,dPjdv;
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

        dEidv.x = 2.0*Adiff*dAidv.x  + 2.0*Pdiff*dPidv.x;
        dEidv.y = 2.0*Adiff*dAidv.y  + 2.0*Pdiff*dPidv.y;
        dEkdv.x = 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x;
        dEkdv.y = 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y;
        dEjdv.x = 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x;
        dEjdv.y = 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y;

        float2 temp = dEidv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;

        float2 temp2 = dEkdv*dhdri[nn];
        forceSum.x += temp2.x;
        forceSum.y += temp2.y;

        float2 temp3 = dEjdv*dhdri[nn];
        forceSum.x += temp3.x;
        forceSum.y += temp3.y;

//        printf("\nvother %i--%i--%i = (%f,%f)\n",baseNeigh,otherNeigh,DT_other_idx,vother.x,vother.y);

//        printf("%f\t %f\t %f\t %f\t %f\t %f\t\n",Adiff,Akdiff,Ajdiff,Pdiff,Pkdiff,Pjdiff);
//        if (i ==0)
//            printf("(%f,%f)\t(%f,%f)\t(%f,%f)\n",dPidv.x,dPidv.y,dPkdv.x,dPkdv.y,dPjdv.x,dPjdv.y);
            //    printf("%f\t%f\t%f\t%f\n",dhdri[nn].x11,dhdri[nn].x12,dhdri[nn].x21,dhdri[nn].x22);
//            printf("%i %f %f\n",nn,temp.x+temp2.x+temp3.x,temp.y+temp2.y+temp3.y);
        vlast=vcur;
        };



    h_f.data[i].x=forceSum.x;
    h_f.data[i].y=forceSum.y;
//    printf("total force on cell: (%f,%f)\n",forceSum.x,forceSum.y);
    };

void SPV2D::computeSPVForceWithTensionsCPU(int i,bool verbose)
    {
    float Pthreshold = 1e-8;
 //   printf("cell %i: \n",i);
    //for testing these routines...
    vector <int> test;
    DelaunayCell celltest;
    delLoc.triangulatePoint(i, test,celltest);

    //read in all the data we'll need
    ArrayHandle<float2> h_p(points,access_location::host,access_mode::read);
    ArrayHandle<float2> h_f(forces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(CellType,access_location::host,access_mode::read);
    ArrayHandle<float2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<float2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(neigh_num,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighs,access_location::host,access_mode::read);

    //get Delaunay neighbors of the cell
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
//        if (i == 602)
//            printf("%i - ",ns[nn]);
        };

    //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
    vector<float2> voro(neigh);
    vector<Matrix2x2> dhdri(neigh);
    Matrix2x2 Id;
    float2 circumcent;
    float2 origin; origin.x = 0.; origin.y=0.;
    float2 rij,rik;
    float2 nnextp,nlastp;
    float2 rjk;
    float2 pi = h_p.data[i];

    nlastp = h_p.data[ns[ns.size()-1]];
    Box.minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        nnextp = h_p.data[ns[nn]];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(origin,rij,rik,circumcent);
        voro[nn] = circumcent;
        rjk.x =rik.x-rij.x;
        rjk.y =rik.y-rij.y;

        float2 dbDdri,dgDdri,dDdriOD,z;
        float betaD = -dot(rik,rik)*dot(rij,rjk);
        float gammaD = dot(rij,rij)*dot(rik,rjk);
        float cp = rij.x*rjk.y - rij.y*rjk.x;
        float D = 2*cp*cp;


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

    float2 vlast,vnext,vother;
    vlast = voro[neigh-1];

    //start calculating forces
    float2 forceSum;
    forceSum.x=0.0;forceSum.y=0.0;
    float KA = 1.0;
    float KP = 1.0;

    float Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
    float Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

//    printf("cell %i: %f, %f\n",i,h_AP.data[i].x,h_APpref.data[i].x);

    float2 vcur;
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


        float2 dEidv,dAidv,dPidv,dTidv;
        dTidv.x = 0.0; dTidv.y = 0.0;

        dAidv.x = 0.5*(vlast.y-vnext.y);
        dAidv.y = 0.5*(vnext.x-vlast.x);

        float2 dlast,dnext;
        dlast.x = vlast.x-vcur.x;
        dlast.y=vlast.y-vcur.y;

        float dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

        dnext.x = vcur.x-vnext.x;
        dnext.y = vcur.y-vnext.y;
        float dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
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
        //
        //now let's compute the other terms...first we need to find the third voronoi
        //position that v_cur is connected to
        //
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
        float2 nl1 = h_p.data[otherNeigh];
        float2 nn1 = h_p.data[baseNeigh];
        float2 no1 = h_p.data[DT_other_idx];

        float2 r1,r2,r3;
        Box.minDist(nl1,pi,r1);
        Box.minDist(nn1,pi,r2);
        Box.minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        float Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        float Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        float Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        float Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        float2 dEkdv,dAkdv,dPkdv,dTkdv;
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

        float2 dEjdv,dAjdv,dPjdv,dTjdv;
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


        dEidv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x + gamma*dTidv.x;
        dEidv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y + gamma*dTidv.y;

        dEkdv.x = 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x + gamma*dTkdv.x;
        dEkdv.y = 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y + gamma*dTkdv.y;

        dEjdv.x = 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x + gamma*dTjdv.x;
        dEjdv.y = 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y + gamma*dTjdv.y;

        float2 temp = dEidv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;

        temp = dEkdv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;

        temp = dEjdv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;

        if(verbose)
            {
            printf("idx %i; baseNeigh %i; otherNeigh %i; DT_other_idx %i\n",i,baseNeigh,otherNeigh,DT_other_idx);
            printf("%f  %f   %f   \n%f  %f    %f\n   %f    %f    %f\n",dAidv.x,dPidv.x,dTidv.x,dAkdv.x, dPkdv.x,dTkdv.x,dAjdv.x,dPjdv.x,dTjdv.x);
            };
//        printf("\nvother %i--%i--%i = (%f,%f)\n",baseNeigh,otherNeigh,DT_other_idx,vother.x,vother.y);

//        printf("%f\t %f\t %f\t %f\t %f\t %f\t\n",Adiff,Akdiff,Ajdiff,Pdiff,Pkdiff,Pjdiff);
        vlast=vcur;
        };



    h_f.data[i].x=forceSum.x;
    h_f.data[i].y=forceSum.y;
//    printf("total force on cell: (%f,%f)\n",forceSum.x,forceSum.y);
    };


void SPV2D::meanArea()
    {
    ArrayHandle<float2> h_AP(AreaPeri,access_location::host,access_mode::read);
    float fx = 0.0;
    for (int i = 0; i < N; ++i)
        {
//        printf("cell %i Area %f\n",i,h_AP.data[i].x);
        fx += h_AP.data[i].x/N;
        };
    printf("Mean area = %f\n" ,fx);

    };


void SPV2D::reportForces()
    {
    ArrayHandle<float2> h_f(forces,access_location::host,access_mode::read);
    ArrayHandle<float2> p(points,access_location::host,access_mode::read);
    float fx = 0.0;
    float fy = 0.0;
    for (int i = 0; i < N; ++i)
        {
        fx += h_f.data[i].x;
        fy += h_f.data[i].y;
//
//        if (isnan(h_f.data[i].x) || isnan(h_f.data[i].y))
//        if(i == N-1)
          printf("cell %i: \t position (%f,%f)\t force (%f, %f)\n",i,p.data[i].x,p.data[i].y ,h_f.data[i].x,h_f.data[i].y);
        };

    };

void SPV2D::meanForce()
    {
    ArrayHandle<float2> h_f(forces,access_location::host,access_mode::read);
    float fx = 0.0;
    float fy = 0.0;
    for (int i = 0; i < N; ++i)
        {
        fx += h_f.data[i].x;
        fy += h_f.data[i].y;
        };
    printf("Mean force = (%e,%e)\n" ,fx,fy);

    };

float SPV2D::reportq()
    {
    ArrayHandle<float2> h_AP(AreaPeri,access_location::host,access_mode::read);
    float A = 0.0;
    float P = 0.0;
    float q = 0.0;
    for (int i = 0; i < N; ++i)
        {
        A = h_AP.data[i].x;
        P = h_AP.data[i].y;
        q += P / sqrt(A);
        };
    return q/(float)N;
    };

void SPV2D::deltaAngle()
    {
    ArrayHandle<float> h_cd(cellDirectors,access_location::host, access_mode::read);
    ArrayHandle<float> h_cdi(cellDirectors_initial,access_location::host, access_mode::read);
    float dA = 0;
    for (int ii = 0; ii < N; ++ii)
        {
        dA += (h_cd.data[ii]-h_cdi.data[ii])* (h_cd.data[ii]-h_cdi.data[ii]);
        };
    dA/=(float)N;
    printf("timestep, dA^2 = (%i,%f)\n",Timestep,dA);


    };

