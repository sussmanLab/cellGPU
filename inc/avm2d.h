//spv.h
#ifndef AVM_H
#define AVM_H


#include "std_include.h"
#include "curand.h"
#include "curand_kernel.h"
#include "gpuarray.h"
#include "gpubox.h"
#include "indexer.h"
#include "cu_functions.h"

#include "DelaunayCGAL.h"

/*!
A class that implements an active vertex model in 2D. This involves calculating forces on
vertices, moving them around, and updating the topology of the cells according to some criteria.

From the point of view of reusing code this could have been a child of SPV2D, but logically since
the AVM does not refer to an underlying triangulation I have decided to implement it separately.
*/
//!Implement a 2D active vertex model
class AVM2D
    {
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true);

        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;
        //! Cell positions... useful for computing the geometry of cells. At the moment cellPositions just ensures that the origin is enclosed by the vertices of a cell. This is irrelevant in almost all of the code, so an optimization would be to remove this.
        GPUArray<Dscalar2> cellPositions;
        //!An array of angles (relative to \hat{x}) that the cell directors point
        GPUArray<Dscalar> cellDirectors;

        //! VERTEX neighbors of every vertex
        GPUArray<int> vertexNeighbors;
        //! Cell neighbors of every vertex
        GPUArray<int> vertexCellNeighbors;
        //! flags that indicate whether an edge should be GPU-flipped (1) or not (0)
        GPUArray<int> vertexEdgeFlips;

        //!an array containing net force on each vertex
        GPUArray<Dscalar2> vertexForces;
        //!an array containing the three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;
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

        //!Enforce CPU-only operation.
        void setCPU(){GPUcompute = false;};

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set uniform motility
        void setv0Dr(Dscalar _v0,Dscalar _Dr){v0=_v0; Dr = _Dr;};

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!initialize the cuda RNG
        void initializeCurandStates(int gs, int i);

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool initGPU = true);

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(int n);

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
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU
        void testAndPerformT1TransitionsGPU();


        //!Get the cell position from the vertices on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellPositionsGPU();


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
        //!The area and perimeter preferences of each cell
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell
        //!The current area and perimeter of each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell

        //! A flag that determines whether the GPU RNG is the same every time.
        bool Reproducible;

        //!The number of vertices defining each cell
        GPUArray<int> cellVertexNum;
        //!A structure that indexes the vertices defining each cell
        GPUArray<int> cellVertices;
        //!A 2dIndexer for computing where in the GPUArray to look for a given cell's vertices
        Index2D n_idx;
        //!An upper bound for the maximum number of neighbors that any cell has
        int vertexMax;


        //utility functions
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

    //reporting functions
    public:
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
                printf("mean force area = (%g,%g)\n",fx/Nvertices, fy/Nvertices);
                };

        void reportAP()
                {
                ArrayHandle<Dscalar2> ap(AreaPeri,access_location::host,access_mode::read);
                Dscalar vtot= 0.0;
                for (int i = 0; i < Ncells; ++i)
                    {
                    printf("%i: (%f,%f)\n",i,ap.data[i].x,ap.data[i].y);
                    vtot+=ap.data[i].x;
                    };
                printf("total area = %f\n",vtot);
                };
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
    };

#endif
