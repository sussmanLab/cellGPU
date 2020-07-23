#include "DelaunayGPU.h"
#include "DelaunayGPU.cuh"
#include "cellListGPU.cuh"
#include "utilities.cuh"

DelaunayGPU::DelaunayGPU() :
	cellsize(1.10), cListUpdated(false), Ncells(0), NumCircumcircles(0), GPUcompute(false)
    {
Box = make_shared<periodicBoundaries>();
    }

DelaunayGPU::DelaunayGPU(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx)
    {
	initialize(N,maximumNeighborsGuess,cellSize,bx);
	}

//!initialization
void DelaunayGPU::initialize(int N, int maximumNeighborsGuess, double cellSize, PeriodicBoxPtr bx)
    {
    prof.start("initialization");
    Ncells = N;
    NumCircumcircles = 0;
    MaxSize = max(4,maximumNeighborsGuess);
    cellsize=cellSize;
    cListUpdated = false;
    setBox(bx);
    sizeFixlist.resize(1);
    maxOneRingSize.resize(1);
    {
    ArrayHandle<int> ms(maxOneRingSize);
    ms.data[0] = MaxSize;
    }
    resize(MaxSize);

    neighs.resize(Ncells);
    repair.resize(Ncells);
    delGPUcircumcircles.resize(Ncells);
    initializeCellList();
    prof.end("initialization");
    }

//Resize the relevant array for the triangulation
void DelaunayGPU::resize(const int nmax)
    {
    MaxSize=nmax;
    GPUVoroCur.resize(nmax*Ncells);
    GPUDelNeighsPos.resize(nmax*Ncells);
    GPUVoroCurRad.resize(nmax*Ncells);
    GPUPointIndx.resize(nmax*Ncells);
    GPU_idx = Index2D(nmax,Ncells);
    }

void DelaunayGPU::initializeCellList()
	{
	cList.setNp(Ncells);
    cList.setBox(Box);
    cList.setGridSize(cellsize);
    }

/*!
\param bx a periodicBoundaries that the DelaunayLoc object should use in internal computations
*/
void DelaunayGPU::setBox(periodicBoundaries &bx)
    {
    cListUpdated=false;
    Box = make_shared<periodicBoundaries>();
    double b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    if (bx.isBoxSquare())
        Box->setSquare(b11,b22);
    else
        Box->setGeneral(b11,b12,b21,b22);
    };

//sets the bucket lists with the points that they contain to use later in the triangulation
void DelaunayGPU::setList(double csize, GPUArray<double2> &points)
{
    cListUpdated=true;
    if(points.getNumElements()!=Ncells || points.getNumElements()==0)
    {
	    printf("GPU DT: No points for cell lists\n");
            throw std::exception();
    }
    if(GPUcompute)
        cList.computeGPU(points);
    else
        cList.compute(points);
}

//automatically goes thorough the process of updating the points
//and lists to get ready for the triangulation (previous initializaton required!).
void DelaunayGPU::updateList(GPUArray<double2> &points)
    {
    if(Ncells != points.getNumElements())
    	cList.setNp(Ncells);

    if(GPUcompute)
        {
        cList.computeGPU(points);
        cudaError_t code = cudaGetLastError();
        if(code!=cudaSuccess)
            {
            printf("cell list computation GPUassert: %s \n", cudaGetErrorString(code));
            throw std::exception();
            };
        }
    else
        cList.compute(points);
    cListUpdated=true;
    }

//call the triangulation routines on a subset of the total number of points
void DelaunayGPU::locallyRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList,int numberToRepair)
    {
    //setRepair(repairList);
    //size_fixlist = numberToRepair;

    int currentN = points.getNumElements();
    if(cListUpdated==false)
		{
        prof.start("cellList");
        updateList(points);
        prof.end("cellList");
		}
    bool recompute = true;
    while (recompute)
        {
        if(GPUcompute==true)
            {
            voronoiCalcRepairList(points, GPUTriangulation, cellNeighborNum,repairList);
            recompute = computeTriangulationRepairList(points, GPUTriangulation, cellNeighborNum,repairList);
            }
        else
            {
            voronoiCalcRepairList_CPU(points, GPUTriangulation, cellNeighborNum,repairList);
            recompute = computeTriangulationRepairList_CPU(points, GPUTriangulation, cellNeighborNum,repairList);
            }
        if(recompute)
            {
            GPUTriangulation.resize(MaxSize*currentN);
            }
        };
    }

//Main function that does the complete triangulation of all points
void DelaunayGPU::globalDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
	int currentN = points.getNumElements();
	if(currentN==0)
        {
        cout<<"No points in GPU DT"<<endl;
        return;
        }
    if(currentN!=Ncells || GPUTriangulation.getNumElements()!=GPUVoroCur.getNumElements())
		{
        Ncells = currentN;
        MaxSize = GPUTriangulation.getNumElements()/Ncells;
        resize(MaxSize);
        initializeCellList();
        cListUpdated = false;
		}
    if(cListUpdated==false)
		{
        prof.start("cellList");
        updateList(points);
        prof.end("cellList");
		cListUpdated=true;
		}

    size_fixlist=Ncells;
    bool recompute = true;
    while (recompute)
	    {
        prof.start("vorocalc");
        if(GPUcompute==true)
            Voronoi_Calc(points, GPUTriangulation, cellNeighborNum);
        else
            Voronoi_Calc_CPU(points, GPUTriangulation, cellNeighborNum);
        prof.end("vorocalc");
        prof.start("get 1 ring");
        if(GPUcompute==true)
            recompute = get_neighbors(points, GPUTriangulation, cellNeighborNum);
        else
            recompute = get_neighbors_CPU(points, GPUTriangulation, cellNeighborNum);
        prof.end("get 1 ring");
        if(recompute)
            {
            GPUTriangulation.resize(MaxSize*currentN);
            }
        };
    }

////////////////////////////////////////////////////
//////
//////           CPU_routines
//////
////////////////////////////////////////////////////

void DelaunayGPU::voronoiCalcRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
    ArrayHandle<double2> d_pt(points,access_location::host,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::host,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::host,access_mode::readwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);
    ArrayHandle<int> d_repair(repairList,access_location::host,access_mode::read);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::host,access_mode::readwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::host,access_mode::readwrite);

    gpu_voronoi_calc_no_sort(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        d_repair.data,
                        GPU_idx,
			            GPUcompute
                        );
    };

//One of the main functions called by the triangulation.
//This creates a simple convex polygon around each point for triangulation.
//Currently the polygon is created with only four points
void DelaunayGPU::Voronoi_Calc_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<double2> d_pt(points,access_location::host,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::host,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::host,access_mode::overwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::host,access_mode::overwrite);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::host,access_mode::overwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::host,access_mode::overwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::host,access_mode::overwrite);

    gpu_voronoi_calc(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        GPU_idx,
			GPUcompute
                        );

    }

bool DelaunayGPU::computeTriangulationRepairList_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
        {
                ArrayHandle<double2> d_pt(points,access_location::host,access_mode::read);
                ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::host,access_mode::read);
                ArrayHandle<int> d_cell_idx(cList.idxs,access_location::host,access_mode::read);

                ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::host,access_mode::readwrite);
                ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);
                ArrayHandle<int> d_repair(repairList,access_location::host,access_mode::read);

                ArrayHandle<double2> d_Q(GPUVoroCur,access_location::host,access_mode::readwrite);
                ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::host,access_mode::readwrite);
                ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::host,access_mode::readwrite);
                ArrayHandle<int> d_ms(maxOneRingSize, access_location::host,access_mode::readwrite);

                gpu_get_neighbors_no_sort(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                d_repair.data,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize,
				GPUcompute
                                    );
        }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
    }

//The final main function of the triangulation.
//This takes the previous polygon and further updates it to create the final delaunay triangulation
bool DelaunayGPU::get_neighbors_CPU(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
{
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
            {
            ArrayHandle<double2> d_pt(points,access_location::host,access_mode::read);
            ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::host,access_mode::read);
            ArrayHandle<int> d_cell_idx(cList.idxs,access_location::host,access_mode::read);

            ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::host,access_mode::readwrite);
            ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::host,access_mode::readwrite);

            ArrayHandle<double2> d_Q(GPUVoroCur,access_location::host,access_mode::readwrite);
            ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::host,access_mode::readwrite);
            ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::host,access_mode::readwrite);
            ArrayHandle<int> d_ms(maxOneRingSize, access_location::host,access_mode::readwrite);

            gpu_get_neighbors(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize,
				GPUcompute
                            );


            }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
}

/*!
only intended to be used as part of the testAndRepair sequence
*/
void DelaunayGPU::testTriangulationCPU(GPUArray<double2> &points)
    {
    {
    ArrayHandle<int> d_repair(repair,access_location::host,access_mode::overwrite);
    for(int ii = 0; ii < Ncells; ++ii)
        d_repair.data[ii] = -1;
    }
    //access data handles
    ArrayHandle<double2> d_pt(points,access_location::host,access_mode::read);

    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> d_c_idx(cList.idxs,access_location::host,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::host,access_mode::readwrite);

    ArrayHandle<int3> d_ccs(delGPUcircumcircles,access_location::host,access_mode::read);

    NumCircumcircles = Ncells*2;
    gpu_test_circumcircles(d_repair.data,
                           d_ccs.data,
                           NumCircumcircles,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           Ncells,
                           cList.getXsize(),
                           cList.getYsize(),
                           cList.getBoxsize(),
                           *(Box),
                           cList.cell_indexer,
                           cList.cell_list_indexer,
                           GPUcompute
                           );
    };


////////////////////////////////////////////////////
//////
//////           GPU_routines
//////
////////////////////////////////////////////////////


void DelaunayGPU::voronoiCalcRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_repair(repairList,access_location::device,access_mode::read);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);

    gpu_voronoi_calc_no_sort(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        d_repair.data,
                        GPU_idx,
                        GPUcompute
                        );
    };

//One of the main functions called by the triangulation.
//This creates a simple convex polygon around each point for triangulation.
//Currently the polygon is created with only four points
void DelaunayGPU::Voronoi_Calc(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::overwrite);
    ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::overwrite);

    ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::overwrite);
    ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::overwrite);
    ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::overwrite);

    gpu_voronoi_calc(d_pt.data,
                        d_cell_sizes.data,
                        d_cell_idx.data,
                        d_P_idx.data,
                        d_P.data,
                        d_Q.data,
                        d_Q_rad.data,
                        d_neighnum.data,
                        Ncells,
                        cList.getXsize(),
                        cList.getYsize(),
                        cList.getBoxsize(),
                        *(Box),
                        cList.cell_indexer,
                        cList.cell_list_indexer,
                        GPU_idx,
                        GPUcompute
                        );


    }

bool DelaunayGPU::computeTriangulationRepairList(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum,GPUArray<int> &repairList)
    {
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
        {
                ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
                ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
                ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

                ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_repair(repairList,access_location::device,access_mode::read);

                ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
                ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
                ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);
                ArrayHandle<int> d_ms(maxOneRingSize, access_location::device,access_mode::readwrite);


                gpu_get_neighbors_no_sort(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                d_repair.data,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize
                                ,GPUcompute
                                );
        }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
    }

//The final main function of the triangulation.
//This takes the previous polygon and further updates it to create the final delaunay triangulation
bool DelaunayGPU::get_neighbors(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
{
        bool recomputeNeighbors = false;
        int postCallMaxOneRingSize;
        int currentMaxOneRingSize = MaxSize;
            {
            ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);
            ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
            ArrayHandle<int> d_cell_idx(cList.idxs,access_location::device,access_mode::read);

            ArrayHandle<int> d_P_idx(GPUTriangulation,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_neighnum(cellNeighborNum,access_location::device,access_mode::readwrite);

            ArrayHandle<double2> d_Q(GPUVoroCur,access_location::device,access_mode::readwrite);
            ArrayHandle<double2> d_P(GPUDelNeighsPos,access_location::device,access_mode::readwrite);
            ArrayHandle<double> d_Q_rad(GPUVoroCurRad,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_ms(maxOneRingSize, access_location::device,access_mode::readwrite);


            gpu_get_neighbors(d_pt.data,
                                d_cell_sizes.data,
                                d_cell_idx.data,
                                d_P_idx.data,
                                d_P.data,
                                d_Q.data,
                                d_Q_rad.data,
                                d_neighnum.data,
                                Ncells,
                                cList.getXsize(),
                                cList.getYsize(),
                                cList.getBoxsize(),
                                *(Box),
                                cList.cell_indexer,
                                cList.cell_list_indexer,
                                GPU_idx,
                                d_ms.data,
                                currentMaxOneRingSize,
                                GPUcompute
                            );
            }
        if(safetyMode)
        {
                {//check initial maximum ring_size allocated
                        ArrayHandle<int> h_ms(maxOneRingSize, access_location::host,access_mode::read);
                        postCallMaxOneRingSize = h_ms.data[0];
                }
                //printf("initial and post %i %i\n", currentMaxOneRingSize,postCallMaxOneRingSize);
                if(postCallMaxOneRingSize > currentMaxOneRingSize)
                {
                        recomputeNeighbors = true;
                        printf("resizing potential neighbors from %i to %i and re-computing...\n",currentMaxOneRingSize,postCallMaxOneRingSize);
                        resize(postCallMaxOneRingSize);
                }
        };
        return recomputeNeighbors;
}

void DelaunayGPU::testAndRepairDelaunayTriangulation(GPUArray<double2> &points, GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    //resize circumcircles array if needed and populate:
    if(delGPUcircumcircles.getNumElements()!= 2*points.getNumElements())
        delGPUcircumcircles.resize(2*points.getNumElements());
    prof.start("getCCS");
    if(GPUcompute)
        getCircumcirclesGPU(GPUTriangulation,cellNeighborNum);
    else
        getCircumcirclesCPU(GPUTriangulation,cellNeighborNum);

    prof.end("getCCS");

    prof.start("cellList");
    if(GPUcompute)
	    cList.computeGPU(points);
    else
	    cList.compute(points);
    cListUpdated=true;
    prof.end("cellList");

    prof.start("testCCS");
    if(GPUcompute)
        testTriangulation(points);
    else
        testTriangulationCPU(points);
    prof.end("testCCS");
    
    //locally repair
    prof.start("repairPoints");
    locallyRepairDelaunayTriangulation(points,GPUTriangulation,cellNeighborNum,repair);
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    prof.end("repairPoints");
    }

/*!
only intended to be used as part of the testAndRepair sequence
*/
void DelaunayGPU::testTriangulation(GPUArray<double2> &points)
    {
    {
    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);
    gpu_set_array(d_repair.data,-1,Ncells);
    }
    //access data handles
    ArrayHandle<double2> d_pt(points,access_location::device,access_mode::read);

    ArrayHandle<unsigned int> d_cell_sizes(cList.cell_sizes,access_location::device,access_mode::read);
    ArrayHandle<int> d_c_idx(cList.idxs,access_location::device,access_mode::read);

    ArrayHandle<int> d_repair(repair,access_location::device,access_mode::readwrite);

    ArrayHandle<int3> d_ccs(delGPUcircumcircles,access_location::device,access_mode::read);

    NumCircumcircles = Ncells*2;
    gpu_test_circumcircles(d_repair.data,
                           d_ccs.data,
                           NumCircumcircles,
                           d_pt.data,
                           d_cell_sizes.data,
                           d_c_idx.data,
                           Ncells,
                           cList.getXsize(),
                           cList.getYsize(),
                           cList.getBoxsize(),
                           *(Box),
                           cList.cell_indexer,
                           cList.cell_list_indexer,
                           GPUcompute
                           );
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    };

void DelaunayGPU::getCircumcirclesGPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<int> assist(sizeFixlist,access_location::device,access_mode::readwrite);
    gpu_set_array(assist.data,0,1);//set the fixlist to zero, for indexing purposes
    ArrayHandle<int> neighbors(GPUTriangulation,access_location::device,access_mode::read);
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<int3> ccs(delGPUcircumcircles,access_location::device,access_mode::overwrite);
    gpu_get_circumcircles(neighbors.data,
                          neighnum.data,
                          ccs.data,
                          assist.data,
                          Ncells,
                          GPU_idx);
#ifdef DEBUGFLAGUP
cudaDeviceSynchronize();
#endif
    }

/*!
Converts the neighbor list data structure into a list of the three particle indices defining
all of the circumcircles in the triangulation. Keeping this version of the topology on the GPU
allows for fast testing of what points need to be retriangulated.
*/
void DelaunayGPU::getCircumcirclesCPU(GPUArray<int> &GPUTriangulation, GPUArray<int> &cellNeighborNum)
    {
    ArrayHandle<int> neighnum(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(GPUTriangulation,access_location::host,access_mode::read);
    ArrayHandle<int3> h_ccs(delGPUcircumcircles,access_location::host,access_mode::overwrite);

    int totaln = 0;
    int cidx = 0;
    int3 cc;
    for (int nn = 0; nn < Ncells; ++nn)
        {
        cc.x = nn;
        int nmax = neighnum.data[nn];
        totaln+=nmax;
        cc.y = ns.data[GPU_idx(nmax-1,nn)];
        for (int jj = 0; jj < nmax; ++jj)
            {
            cc.z = ns.data[GPU_idx(jj,nn)];
            if (cc.x < cc.y && cc.x < cc.z)
                {
                h_ccs.data[cidx] = cc;
                cidx+=1;
                }
            cc.y = cc.z;
            };
        };
    NumCircumcircles = cidx;

    if((totaln != 6*Ncells || cidx != 2*Ncells))
        {
        printf("GPU step: getCCs failed, %i out of %i ccs, %i out of %i neighs \n",cidx,2*Ncells,totaln,6*Ncells);
        throw std::exception();
        };

    };
