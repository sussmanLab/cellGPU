#ifndef AVM_H
#define AVM_H

#include "std_include.h"
#include "gpuarray.h"
#include "indexer.h"
#include "cu_functions.h"
#include "Simple2DCell.h"

#include <cuda.h>

/*!
A class that implements an active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.

From the point of view of reusing code this could have been (should have been?)a child of SPV2D,
but logically since the AVM does not refer to an underlying triangulation I have decided to
implement it as a separate class.
*/
//!Implement a 2D active vertex model, using kernels in \ref avmKernels
class AVM2D : public Simple2DCell
    {
    //public functions first... many of these should eventually be protected, but for debugging
    //it's convenient to be able to call them from anywhere
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true,bool runSPVToInitialize=false);

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool initGPU = true,bool spvInitialize = false);

        //!Set uniform motility
        void setv0Dr(Dscalar _v0,Dscalar _Dr){v0=_v0; Dr = _Dr;};

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!Set the length threshold for T1 transitions
        void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};

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
        */
        //!3*Nvertices length array of the position of vertices around cells
        GPUArray<Dscalar2> voroCur;
        //!3*Nvertices length array of the position of the last and next vertices along the cell
        GPUArray<Dscalar4> voroLastNext;

        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

    //protected variables
    protected:
        //!the area modulus
        Dscalar KA;
        //!The perimeter modulus
        Dscalar KP;

        //!velocity of vertices
        Dscalar v0;
        //!rotational diffusion of vertex directors
        Dscalar Dr;

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

        //! it is important to not flip edges concurrently, so this data structure helps flip edges sequentially
        GPUArray<int> vertexEdgeFlipsCurrent;
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
