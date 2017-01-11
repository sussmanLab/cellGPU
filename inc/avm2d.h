#ifndef AVM_H
#define AVM_H

#include "std_include.h"
#include "curand.h"
#include "curand_kernel.h"
#include "gpuarray.h"
#include "gpubox.h"
#include "indexer.h"
#include "cu_functions.h"

#include <cuda.h>

/*!
A class that implements an active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.

From the point of view of reusing code this could have been (should have been?)a child of SPV2D,
but logically since the AVM does not refer to an underlying triangulation I have decided to
implement it as a separate class.
*/
//!Implement a 2D active vertex model, using kernels in \ref avmKernels
class AVM2D
    {
    //public functions first... many of these should eventually be protected, but for debugging
    //it's convenient to be able to call them from anywher
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true,bool runSPVToInitialize=false);
        //!Enforce CPU-only operation.
        void setCPU(){GPUcompute = false;};

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool initGPU = true,bool spvInitialize = false);

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set uniform motility
        void setv0Dr(Dscalar _v0,Dscalar _Dr){v0=_v0; Dr = _Dr;};

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!Set the length threshold for T1 transitions
        void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};

        //!initialize the cuda RNG
        void initializeCurandStates(int gs, int i);

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(int n, bool spvInitialize = false);

        //!if the maximum number of vertices per cell increases, grow the cellVertices list
        void growCellVerticesList(int newVertexMax);

        //!progress through the parts of a time step...simply an interface to the correct other procedure
        void performTimestep();
        //!progress through the parts of a time step on the CPU
        void performTimestepCPU();
        //!progress through the parts of a time step on the GPU
        void performTimestepGPU();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeGeometryGPU();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeForcesCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeForcesGPU();

        //!Displace vertices and rotate directors on the CPU
        void displaceAndRotateCPU();
        //!Displace vertices and rotate directors on the GPU
        void displaceAndRotateGPU();

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();
        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();


        //!Get the cell position from the vertices on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellPositionsGPU();

    //protected functions
    protected:
        //utility functions
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

    //public member variables...most of these should eventually be protected
    public:
        //! Cell positions... not used for computation, but can track, e.g., MSD of cell centers
        GPUArray<Dscalar2> cellPositions;
        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;

        /*!
        vertexNeighbors[3*i], vertexNeighbors[3*i+1], and vertexNeighbors[3*i+2] contain the indices
        of the three vertices that are connected to vertex i
        */
        //! VERTEX neighbors of every vertex
        GPUArray<int> vertexNeighbors;
        /*!
        vertexCellNeighbors[3*i], vertexCellNeighbors[3*i+1], and vertexCellNeighbors[3*i+2] contain
        the indices of the three cells are niehgbors of vertex i
        */
        //! Cell neighbors of every vertex
        GPUArray<int> vertexCellNeighbors;
        /*!
        if vertexEdgeFlips[3*i+j]=1 (where j runs from 0 to 2), the the edge connecting verte i and vertex
        vertexNeighbors[3*i+j] has been marked for a T1 transition
        */
        //!An array of angles (relative to \hat{x}) that the cell directors point
        GPUArray<Dscalar> cellDirectors;
        //! flags that indicate whether an edge should be GPU-flipped (1) or not (0)
        GPUArray<int> vertexEdgeFlips;
        //! it is important to not flip edges concurrently, so this data structure helps flip edges sequentially
        GPUArray<int> vertexEdgeFlipsCurrent;

        //!an array containing net force on each vertex
        GPUArray<Dscalar2> vertexForces;
        /*!
        vertexForceSets[3*i], vertexForceSets[3*i+1], and vertexForceSets[3*i+2] contain the contribution
        to the net force on vertex i due to the three cell neighbors of vertex i
        */
        //!an array containing the three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;
        /*!
        when computing the geometry of the cells, save the relative position of the vertices for easier force calculation later
        The "voro" part is an unfortunate naming holdover from the SPV code, where they are actually Voronoi vertices
        */
        //!3*Nvertices length array of the position of voro vertex
        GPUArray<Dscalar2> voroCur;
        //!3*Nvertices length array of the position of the last and next voro vertices along the cell
        GPUArray<Dscalar4> voroLastNext;

        //! Count the number of times "performTimeStep" has been called
        int Timestep;

        //!The time stepsize of the simulation
        Dscalar deltaT;

        //!the box defining the periodic domain
        gpubox Box;

        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

    //protected variables
    protected:
        //!Number of cells in the simulation
        int Ncells;
        //!Number of vertices (i.e, degrees of freedom)
        int Nvertices;

        //!A flag that, when true, has performTimestep call the GPU routines
        bool GPUcompute;

        //!the area modulus
        Dscalar KA;
        //!The perimeter modulus
        Dscalar KP;

        //!velocity of vertices
        Dscalar v0;
        //!rotational diffusion of vertex directors
        Dscalar Dr;

        //!A flag to determine whether the CUDA RNGs should be initialized or not (so that the program will run on systems with no GPU by setting this to false
        bool initializeGPURNG;
        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> devStates;
        //!The current area and perimeter of each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell
        //!The area and perimeter preferences of each cell
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell

        //! A flag that determines whether the GPU RNG is the same every time.
        bool Reproducible;

        /*!
        cellVertexNum[c] is an integer storing the number of vertices that make up the boundary of cell c.
        */
        //!The number of vertices defining each cell
        GPUArray<int> cellVertexNum;
        /*!
        cellVertices is a large, 1D array containing the vertices associated with each cell.
        It must be accessed with the help of the Index2D structure n_idx.
        the index of the kth vertex of cell c (where the ordering is counter-clockwise starting
        with a random vertex) is given by
        cellVertices[n_idx(k,c)];
        */
        //!A structure that indexes the vertices defining each cell
        GPUArray<int> cellVertices;
        //!A 2dIndexer for computing where in the GPUArray to look for a given cell's vertices
        Index2D n_idx;
        //!An upper bound for the maximum number of neighbors that any cell has
        int vertexMax;

        //! data structure to help with cell-vertex list
        GPUArray<int> growCellVertexListAssist;

        //! data structure to help with not simultaneously trying to flip nearby edges
        GPUArray<int> finishedFlippingEdges;

        //! A locking structure to allow mutex in kernels
//        Lock myLock;


    //reporting functions
    public:
        //!Report the current average force per vertex...should be close to zero
        void reportMeanForce()
                {
                ArrayHandle<Dscalar2> f(vertexForces,access_location::host,access_mode::read);
                Dscalar fx= 0.0;
                Dscalar fy = 0.0;
                for (int i = 0; i < Nvertices; ++i)
                    {
                    fx += f.data[i].x;
                    fy += f.data[i].y;
                    };
                printf("mean force = (%g,%g)\n",fx/Nvertices, fy/Nvertices);
                };

        //!report the current total area, and optionally the area and perimeter for each cell
        void reportAP(bool verbose = false)
                {
                ArrayHandle<Dscalar2> ap(AreaPeri,access_location::host,access_mode::read);
                Dscalar vtot= 0.0;
                for (int i = 0; i < Ncells; ++i)
                    {
                    if(verbose)
                        printf("%i: (%f,%f)\n",i,ap.data[i].x,ap.data[i].y);
                    vtot+=ap.data[i].x;
                    };
                printf("total area = %f\n",vtot);
                };

        //!Handy for debugging T1 transitions...report the vertices owned by cell i
        void reportNeighborsCell(int i)
            {
            ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
            ArrayHandle<int> h_cv(cellVertices,access_location::host,access_mode::read);
            int cn = h_cvn.data[i];
            printf("Cell %i's neighbors:\n",i);
            for (int n = 0; n < cn; ++n)
                {
                printf("%i, ",h_cv.data[n_idx(n,i)]);
                }
            cout <<endl;
            };
    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabase;
    };
#endif
