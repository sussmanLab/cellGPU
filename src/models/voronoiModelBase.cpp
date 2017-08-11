#define ENABLE_CUDA

#include "cuda_runtime.h"
#include "voronoiModelBase.h"
#include "voronoiModelBase.cuh"

/*! \file voronoiModelBase.cpp */

/*!
A simple constructor that sets many of the class member variables to zero
*/
voronoiModelBase::voronoiModelBase() :
    cellsize(1.25), timestep(0),repPerFrame(0.0),skippedFrames(0),
    neighMax(0),neighMaxChange(false),GlobalFixes(0),globalOnly(true)
    {
    //set cellsize to about unity...magic number should be of order 1
    //when the box area is of order N (i.e. on average one particle per bin)

    };

/*!
 * a function that takes care of the initialization of the class.
 * \param n the number of cells to initialize
 */
void voronoiModelBase::initializeDelMD(int n)
    {
    GPUcompute = true;

    //set particle number and box
    Ncells = n;
    Dscalar boxsize = sqrt((Dscalar)Ncells);
    Box.setSquare(boxsize,boxsize);

    //set circumcenter array size
    circumcenters.resize(2*(Ncells+10));
    NeighIdxs.resize(6*(Ncells+10));

    cellPositions.resize(Ncells);
    repair.resize(Ncells);

    setCellPositionsRandomly();
    //initialize spatial sorting, but do not sort by default
    initializeCellSorting();

    //cell list initialization
    celllist.setNp(Ncells);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);

    //DelaunayLoc initialization
    gpubox Bx(boxsize,boxsize);
    delLoc.setBox(Bx);
    resetDelLocPoints();

    //make a full triangulation
    completeRetriangulationPerformed = 1;
    cellNeighborNum.resize(Ncells);
    globalTriangulationCGAL();

    //initialize the anyCircumcenterTestFailed structure
    anyCircumcenterTestFailed.resize(1);
    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::overwrite);
    h_actf.data[0]=0;
    };

/*!
The GPU moves the location of points in the GPU memory... this gets a local copy that can be used
by the DelaunayLoc class
\post the DelaunayLoc class has the updated set of cell positions, and its cell list is initialized
*/
void voronoiModelBase::resetDelLocPoints()
    {
    ArrayHandle<Dscalar2> h_points(cellPositions,access_location::host, access_mode::read);
    delLoc.setPoints(h_points,Ncells);
    delLoc.initialize(cellsize);
    };

/*!
\post the cell list is updated according to the current cell positions
*/
void voronoiModelBase::updateCellList()
    {

    if(GPUcompute)
        {
        celllist.computeGPU(cellPositions);
        cudaError_t code = cudaGetLastError();
        if(code!=cudaSuccess)
            {
            printf("cell list computation GPUassert: %s \n", cudaGetErrorString(code));
            throw std::exception();
            };
        }
    else
        {
        vector<Dscalar> psnew(2*Ncells);
        ArrayHandle<Dscalar2> h_points(cellPositions,access_location::host, access_mode::read);
        for (int ii = 0; ii < Ncells; ++ii)
            {
            psnew[2*ii] =  h_points.data[ii].x;
            psnew[2*ii+1]= h_points.data[ii].y;
            };
        celllist.setParticles(psnew);
        celllist.compute();

        };

    };

/*!
Displace cells on the CPU
\param displacements a vector of Dscalar2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void voronoiModelBase::movePointsCPU(GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_d(displacements,access_location::host,access_mode::read);
    for (int idx = 0; idx < Ncells; ++idx)
        {
        h_p.data[idx].x += h_d.data[idx].x;
        h_p.data[idx].y += h_d.data[idx].y;
        Box.putInBoxReal(h_p.data[idx]);
        };
    };

/*!
Displace cells on the GPU
\param displacements a vector of Dscalar2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void voronoiModelBase::movePoints(GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_d(displacements,access_location::device,access_mode::readwrite);
    gpu_move_degrees_of_freedom(d_p.data,d_d.data,Ncells,Box);
    cudaError_t code = cudaGetLastError();
    if(code!=cudaSuccess)
        {
        printf("movePoints GPUassert: %s \n", cudaGetErrorString(code));
        throw std::exception();
        };
    };

/*!
Displace cells on either the GPU or CPU, according to the flag
\param displacements a vector of Dscalar2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void voronoiModelBase::moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements)
    {
    if (GPUcompute)
        movePoints(displacements);
    else
        movePointsCPU(displacements);
    };

/*!
The DelaunayLoc and DelaunayNP classes are invoked to performed to determine the Delaunay
triangulation of the entire periodic domain.
*/
void voronoiModelBase::fullTriangulation(bool verbose)
    {
    GlobalFixes +=1;
    completeRetriangulationPerformed = 1;
    resetDelLocPoints();

    //get neighbors of each cell in CW order

    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);
    vector< vector<int> > allneighs(Ncells);
    int oldNmax = neighMax;
    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < Ncells; ++nn)
        {
        vector<int> neighTemp;
        delLoc.getNeighbors(nn,neighTemp);
        allneighs[nn]=neighTemp;
        neighnum.data[nn] = neighTemp.size();
        totaln += neighTemp.size();
        if (neighTemp.size() > nmax) nmax= neighTemp.size();
        h_repair.data[nn]=0;
        };
    if (nmax%2 ==0)
        neighMax = nmax + 2;
    else
        neighMax = nmax + 1;

    n_idx = Index2D(neighMax,Ncells);
    if(neighMax != oldNmax)
        {
        cellNeighbors.resize(neighMax*Ncells);
        neighMaxChange = true;
        };
    updateNeighIdxs();


    //store data in gpuarray
    {
    ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::overwrite);
    for (int nn = 0; nn < Ncells; ++nn)
        {
        int imax = neighnum.data[nn];
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,nn);
            ns.data[idxpos] = allneighs[nn][ii];
            };
        };

    if(verbose)
        cout << "global new Nmax = " << neighMax << "; total neighbors = " << totaln << endl;cout.flush();
    };

    getCircumcenterIndices();

    if(totaln != 6*Ncells)
        {
        printf("CPU neighbor creation failed to match topology! NN = %i \n",totaln);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        getCircumcenterIndices();
        writeTriangulation(output);
        throw std::exception();
        };
    };

/*!
This function calls the DelaunayCGAL class to determine the Delaunay triangulation of the entire
square periodic domain this method is, obviously, better than the version written by DMS, so
should be the default option. In addition to performing a triangulation, the function also automatically
calls updateNeighIdxs and getCircumcenterIndices/
*/
void voronoiModelBase::globalTriangulationCGAL(bool verbose)
    {
    GlobalFixes +=1;
    completeRetriangulationPerformed = 1;
    DelaunayCGAL dcgal;
    ArrayHandle<Dscalar2> h_points(cellPositions,access_location::host, access_mode::read);
    vector<pair<Point,int> > Psnew(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Psnew[ii]=make_pair(Point(h_points.data[ii].x,h_points.data[ii].y),ii);
        };
    Dscalar b1,b2,b3,b4;
    Box.getBoxDims(b1,b2,b3,b4);
    dcgal.PeriodicTriangulation(Psnew,b1);

    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);

    int oldNmax = neighMax;
    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < Ncells; ++nn)
        {
        neighnum.data[nn] = dcgal.allneighs[nn].size();
        totaln += dcgal.allneighs[nn].size();
        if (dcgal.allneighs[nn].size() > nmax) nmax= dcgal.allneighs[nn].size();
        h_repair.data[nn]=0;
        };
    if (nmax%2 == 0)
        neighMax = nmax+2;
    else
        neighMax = nmax+1;

    n_idx = Index2D(neighMax,Ncells);
    if(neighMax != oldNmax)
        {
        cellNeighbors.resize(neighMax*Ncells);
        neighMaxChange = true;
        };
    updateNeighIdxs();

    //store data in gpuarrays
    {
    ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::overwrite);

    for (int nn = 0; nn < Ncells; ++nn)
        {
        int imax = neighnum.data[nn];
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,nn);
            ns.data[idxpos] = dcgal.allneighs[nn][ii];
            };
        };

    if(verbose)
        cout << "global new Nmax = " << neighMax << "; total neighbors = " << totaln << endl;cout.flush();
    };

    getCircumcenterIndices(true);

    if(totaln != 6*Ncells)
        {
        printf("global CPU neighbor failed! NN = %i\n",totaln);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        throw std::exception();
        };
    };

/*!
\post the NeighIdx data structure is updated, which helps cut down on the number of inactive
threads in the force set computation function
*/
void voronoiModelBase::updateNeighIdxs()
    {
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int2> h_nidx(NeighIdxs,access_location::host,access_mode::overwrite);
    int idx = 0;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int nmax = neighnum.data[ii];
        for (int nn = 0; nn < nmax; ++nn)
            {
            h_nidx.data[idx].x = ii;
            h_nidx.data[idx].y = nn;
            idx+=1;
            };
        };
    NeighIdxNum = idx;
    };

/*!
Converts the neighbor list data structure into a list of the three particle indices defining
all of the circumcenters in the triangulation. Keeping this version of the topology on the GPU
allows for fast testing of what points need to be retriangulated.
*/
void voronoiModelBase::getCircumcenterIndices(bool secondtime, bool verbose)
    {
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int3> h_ccs(circumcenters,access_location::host,access_mode::overwrite);

    int totaln = 0;
    int cidx = 0;
    bool fail = false;
    for (int nn = 0; nn < Ncells; ++nn)
        {
        int nmax = neighnum.data[nn];
        totaln+=nmax;
        for (int jj = 0; jj < nmax; ++jj)
            {
            if (fail) continue;

            int n1 = ns.data[n_idx(jj,nn)];
            int ne2 = jj + 1;
            if (jj == nmax-1)  ne2=0;
            int n2 = ns.data[n_idx(ne2,nn)];
            if (nn < n1 && nn < n2)
                {
                h_ccs.data[cidx].x = nn;
                h_ccs.data[cidx].y = n1;
                h_ccs.data[cidx].z = n2;
                cidx+=1;
                };
            };
        };
    NumCircumCenters = cidx;
    if((totaln != 6*Ncells || cidx != 2*Ncells) && !secondtime)
        globalTriangulationCGAL();
    if((totaln != 6*Ncells || cidx != 2*Ncells) && secondtime)
        {
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        printf("step: %i  getCCs failed, %i out of %i ccs, %i out of %i neighs \n",timestep,cidx,2*Ncells,totaln,6*Ncells);
        globalTriangulationCGAL();
        throw std::exception();
        };
    };

/*!
Given a list of particle indices that need to be repaired, call CGAL to figure out their neighbors
and then update the relevant data structures.
*/
void voronoiModelBase::repairTriangulation(vector<int> &fixlist)
    {
    int fixes = fixlist.size();
    repPerFrame += ((Dscalar) fixes/(Dscalar)Ncells);
    resetDelLocPoints();

    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);

    //First, retriangulate the target points, and check if the neighbor list needs to be reset
    //the structure you want is vector<vector<int> > allneighs(fixes), but below a flattened version is implemented

    vector<int> allneighs;
    allneighs.reserve(fixes*neighMax);
    vector<int> allneighidxstart(fixes);
    vector<int> allneighidxstop(fixes);

    //vector<vector<int> > allneighs(fixes);
    vector<int> neighTemp;
    neighTemp.reserve(10);

    bool resetCCidx = false;
    bool LocalFailure = false;
    bool localTest = false;
    for (int ii = 0; ii < fixes; ++ii)
        {
        int pidx = fixlist[ii];
        localTest = delLoc.getNeighborsCGAL(pidx,neighTemp);
        if(!localTest)
            {
            LocalFailure = true;
            cout << "local triangulation failed...attempting a global triangulation to save the day" << endl << "Note that a particle position has probably become NaN, in which case CGAL will give an assertion violation" << endl;
            break;
            };

        allneighidxstart[ii] = allneighs.size();
        for (int nn = 0; nn < neighTemp.size(); ++nn)
            {
            allneighs.push_back(neighTemp[nn]);
            };
        //allneighs[ii]=neighTemp;

        allneighidxstop[ii] = allneighs.size();
        if(neighTemp.size() > neighMax)
            {
            resetCCidx = true;
            };
        };

    //if needed, regenerate the "neighs" structure...hopefully don't do this too much
    if(resetCCidx || LocalFailure)
        {
        if(resetCCidx)
            neighMaxChange = true;
        globalTriangulationCGAL();
        return;
        };

    //now, edit the right entries of the neighborlist and neighbor size list
    ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::readwrite);
    for (int nn = 0; nn < fixes; ++nn)
        {
        int pidx = fixlist[nn];
        //int imax = allneighs[nn].size();
        int imax = allneighidxstop[nn]-allneighidxstart[nn];
        neighnum.data[pidx] = imax;
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,pidx);
            //ns.data[idxpos] = allneighs[nn][ii];
            ns.data[idxpos] = allneighs[ii+allneighidxstart[nn]];
            };
        };

    //finally, update the NeighIdx list and Circumcenter list
    updateNeighIdxs();
    getCircumcenterIndices();
    };

/*!
Call the GPU to test each circumcenter to see if it is still empty (i.e., how much of the
triangulation from the last time step is still valid?). Note that because gpu_test_circumcenters
*always* copies at least a single integer back and forth (to answer the question "did any
circumcircle come back non-empty?" for the cpu) this function is always an implicit cuda
synchronization event. At least until non-default streams are added to the code.
*/
void voronoiModelBase::testTriangulation()
    {
    //first, update the cell list, and set the cc test to 0
    updateCellList();
    {
    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::overwrite);
    h_actf.data[0]=0;
    };
    //access data handles
    ArrayHandle<Dscalar2> d_pt(cellPositions,access_location::device,access_mode::read);

    ArrayHandle<unsigned int> d_cell_sizes(celllist.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(celllist.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
    ArrayHandle<int3> d_ccs(circumcenters,access_location::device,access_mode::read);
    ArrayHandle<int> d_actf(anyCircumcenterTestFailed,access_location::device,access_mode::readwrite);

    int NumCircumCenters = Ncells*2;
    gpu_test_circumcenters(d_repair.data,
                           d_ccs.data,
                           NumCircumCenters,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           Ncells,
                           celllist.getXsize(),
                           celllist.getYsize(),
                           celllist.getBoxsize(),
                           Box,
                           celllist.cell_indexer,
                           celllist.cell_list_indexer,
                           d_actf.data
                           );
    };

/*!
perform the same check on the CPU... because of the cost of checking circumcircles and the
relatively poor performance of the 1-ring calculation in DelaunayLoc, it is sometimes better
to just re-triangulate the entire point set with CGAL. At the moment that is the default
behavior of the cpu branch.
*/
void voronoiModelBase::testTriangulationCPU()
    {
    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::readwrite);
    h_actf.data[0]=0;
    if (globalOnly)
        {
        globalTriangulationCGAL();
        skippedFrames -= 1;
        }
    else
        {
        resetDelLocPoints();

        ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
        ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);
        ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::readwrite);
        h_actf.data[0]=0;
        for (int nn = 0; nn < Ncells; ++nn)
            {
            h_repair.data[nn] = 0;
            vector<int> neighbors;
            neighbors.reserve(neighMax);
            for (int ii = 0; ii < neighnum.data[nn];++ii)
                    {
                    int idxpos = n_idx(ii,nn);
                    neighbors.push_back(ns.data[idxpos]);
                    };

            bool good = delLoc.testPointTriangulation(nn,neighbors,false);
            if(!good)
                {
                h_repair.data[nn] = 1;
                h_actf.data[0]=1;
                };
            };
        };
    };

/*!
This function calls the relevant testing and repairing functions, and increments the "timestep"
by one. Note that the call to testTriangulation will always synchronize the gpu (via a memcpy of
the "anyCircumcenterTestFailed" variable)
*/
void voronoiModelBase::testAndRepairTriangulation(bool verb)
    {
    timestep +=1;

    if (verb) printf("testing triangulation\n");
    if(GPUcompute)
        {
        testTriangulation();
        }
    else
        {
        testTriangulationCPU();
        };

    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::read);
    if(h_actf.data[0]==1)
        {
        NeedsFixing.clear();
        ArrayHandle<int> h_repair(repair,access_location::host,access_mode::readwrite);
        if(GPUcompute)
            {
            cudaError_t code = cudaGetLastError();
            if(code!=cudaSuccess)
                {
                printf("testAndRepair preliminary GPUassert: %s \n", cudaGetErrorString(code));
                throw std::exception();
                };
            };

        //add the index and all of its' neighbors
        ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);
        ArrayHandle<int> ns(cellNeighbors,access_location::host,access_mode::readwrite);
        for (int nn = 0; nn < Ncells; ++nn)
            {
            if (h_repair.data[nn] == 1)
                {
                NeedsFixing.push_back(nn);
                h_repair.data[nn] = 0;
                for (int ii = 0; ii < neighnum.data[nn];++ii)
                    {
                    int idxpos = n_idx(ii,nn);
                    NeedsFixing.push_back(ns.data[idxpos]);
                    };
                };
            };
        sort(NeedsFixing.begin(),NeedsFixing.end());
        NeedsFixing.erase(unique(NeedsFixing.begin(),NeedsFixing.end() ),NeedsFixing.end() );

        if (verb) printf("repairing triangulation via %lu\n",NeedsFixing.size());

        if (NeedsFixing.size() > (Ncells/6))
            {
            completeRetriangulationPerformed = 1;
            globalTriangulationCGAL();
            }
        else
            {
            completeRetriangulationPerformed = 0;
            repairTriangulation(NeedsFixing);
            };
        }
    else
        skippedFrames+=1;
    };

//read a triangulation from a text file...used only for testing purposes. Any other use should call the Database class (see inc/Database.h")
void voronoiModelBase::readTriangulation(ifstream &infile)
    {
    string line;
    getline(infile,line);
    stringstream convert(line);
    int nn;
    convert >> nn;
    cout << "Reading in " << nn << "points" << endl;
    int idx = 0;
    int ii = 0;
    ArrayHandle<Dscalar2> p(cellPositions,access_location::host,access_mode::overwrite);
    while(getline(infile,line))
        {
        Dscalar val = stof(line);
        if (idx == 0)
            {
            p.data[ii].x=val;
            idx +=1;
            }
        else
            {
            p.data[ii].y=val;
            Box.putInBoxReal(p.data[ii]);
            idx = 0;
            ii += 1;
            };
        };
    };

//similarly, write a text file with particle positions. This is often called when an exception is thrown
void voronoiModelBase::writeTriangulation(ofstream &outfile)
    {
    ArrayHandle<Dscalar2> p(cellPositions,access_location::host,access_mode::read);
    outfile << Ncells <<endl;
    for (int ii = 0; ii < Ncells ; ++ii)
        outfile << p.data[ii].x <<"\t" <<p.data[ii].y <<endl;
    };

/*!
\pre Topology is up-to-date on the CPU
\post geometry and voronoi neighbor locations are computed for the current configuration
*/
void voronoiModelBase::computeGeometryCPU()
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
\pre The topology of the Delaunay triangulation is up-to-date on the GPU
\post calculate all cell areas, perimenters, and voronoi neighbors
*/
void voronoiModelBase::computeGeometryGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_nn(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(cellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::overwrite);

    gpu_compute_voronoi_geometry(
                        d_p.data,
                        d_AP.data,
                        d_nn.data,
                        d_n.data,
                        d_vc.data,
                        d_vln.data,
                        Ncells, n_idx,Box);
    };

/*!
As the code is modified, all GPUArrays whose size depend on neighMax should be added to this function
\post voroCur,voroLastNext, delSets, delOther, and forceSets grow to size neighMax*Ncells
*/
void voronoiModelBase::resetLists()
    {
    voroCur.resize(neighMax*Ncells);
    voroLastNext.resize(neighMax*Ncells);
    delSets.resize(neighMax*Ncells);
    delOther.resize(neighMax*Ncells);
    forceSets.resize(neighMax*Ncells);
    };

/*!
\param i the cell in question
\post the delSet and delOther data structure for cell i is updated. Recall that
delSet.data[n_idx(nn,i)] is an int2; the x and y parts store the index of the previous and next
Delaunay neighbor, ordered CCW. delOther contains the mutual neighbor of delSet.data[n_idx(nn,i)].y
and delSet.data[n_idx(nn,i)].z that isn't cell i
*/
bool voronoiModelBase::getDelSets(int i)
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
Calls updateNeighIdxs and then getDelSets(i) for all cells i
*/
void voronoiModelBase::allDelSets()
    {
    updateNeighIdxs();
    for (int ii = 0; ii < Ncells; ++ii)
        getDelSets(ii);
    };


/*!
Trigger a cell division event, which involves some laborious re-indexing of various data structures.
parameters[0] = the index of the cell to undergo a division event
dParams[0] = an angle, theta
dParams[1] = a fraction of maximum separation of the new cell positions along the axis of the cell specified by theta
\post This function is meant to be called before the start of a new timestep. It should be immediately followed by a computeGeometry call
*/
void voronoiModelBase::cellDivision(const vector<int> &parameters, const vector<Dscalar> &dParams)
    {
    int cellIdx = parameters[0];
    Dscalar theta = dParams[0];
    Dscalar separationFraction = dParams[1];

    //First let's get the geometry of the cell in a convenient reference frame
    //computeGeometry has not yet been called, so need to find the voro positions
    vector<Dscalar2> voro;
    voro.reserve(10);
    int neigh;
    Dscalar2 initialCellPosition;
    {//arrayHandle scope
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    initialCellPosition = h_p.data[cellIdx];
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    neigh = h_nn.data[cellIdx];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
            ns[nn]=h_n.data[n_idx(nn,cellIdx)];
    Dscalar2 circumcent;
    Dscalar2 nnextp,nlastp;
    Dscalar2 pi = h_p.data[cellIdx];
    Dscalar2 rij, rik;
    nlastp = h_p.data[ns[ns.size()-1]];
    Box.minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        nnextp = h_p.data[ns[nn]];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(rij,rik,circumcent);
        voro.push_back(circumcent);
        rij=rik;
        }
    };//arrayHandle scope

    //find where the line emanating from the polygons intersects the edges
    Dscalar2 c,v1,v2,Int1,Int2;
    bool firstIntFound = false;
    v1 = voro[neigh-1];
    Dscalar2 ray; ray.x = Cos(theta); ray.y = Sin(theta);
    c.x = - Sin(theta);
    c.y = Cos(theta);
    Dscalar2 p; p.x =0.0; p.y=0.;
    for (int nn = 0; nn < neigh; ++nn)
        {
        v2=voro[nn];
        Dscalar2 a; a.x = p.x-v2.x; a.y = p.y-v2.y;
        Dscalar2 b; b.x = v1.x - v2.x; b.y = v1.y-v2.y;
        Dscalar t2 = -1.;
        if (dot(b,c) != 0)
            t2 = dot(a,c)/dot(b,c);
        if (t2 >= 0. && t2 <= 1.0)
            {
            if (firstIntFound)
                {
                Int2 = v2+t2*(v1-v2);
                }
            else
                {
                Int1 = v2+t2*(v1-v2);
                firstIntFound = true;
                };
            };

        v1=v2;
        };
    Dscalar maxSeparation = max(norm(p-Int1),norm(p-Int2));
    Dscalar2 newCellPos1 = initialCellPosition + separationFraction*maxSeparation*ray;
    Dscalar2 newCellPos2 = initialCellPosition - separationFraction*maxSeparation*ray;
    Box.putInBoxReal(newCellPos1);
    Box.putInBoxReal(newCellPos2);

    printf("((%f,%f), (%f,%f), (%f,%f))\n",initialCellPosition.x,initialCellPosition.y,newCellPos1.x,newCellPos1.y,newCellPos2.x,newCellPos2.y);

    //This call updates many of the base data structres, but (among other things) does not actually
    //set the new cell position
    Simple2DActiveCell::cellDivision(parameters);


    /* the initialization routine gives hints of things that need to be updated here
    displacements.resize(n);
    cellForces.resize(n);
    external_forces.resize(n);
    exclusions.resize(n);

    circumcenters.resize(2*(Ncells+10));

    NeighIdxs.resize(6*(Ncells+10));
    repair.resize(Ncells);
    //initialize spatial sorting, but do not sort by default
    initializeCellSorting();
    //cell list initialization
    celllist.setNp(Ncells);
    celllist.setBox(Box);
    celllist.setGridSize(cellsize);
    //DelaunayLoc initialization
    resetDelLocPoints();
    //make a full triangulation
    completeRetriangulationPerformed = 1;
    cellNeighborNum.resize(Ncells);
    globalTriangulationCGAL();
    //initialize the anyCircumcenterTestFailed structure
    anyCircumcenterTestFailed.resize(1);
    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::overwrite);
    h_actf.data[0]=0;
    */
    //resetLists() will take care of voroCur, voroLastNext, delSets, delOther, and forceSets
    //the allDelSets() should be called
    };
