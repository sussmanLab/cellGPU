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
*/
//!Implement a 2D active vertex model
class AVM2D
    {
    public:
        //! the constructor: initialize as a Delaunay configuration with random positions and set all cells to have uniform target A_0 and P_0 parameters
        AVM2D(int n, Dscalar A0, Dscalar P0,bool reprod = false,bool initGPURNG=true);

        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;
        //! Cell positions... useful for computing the geometry of cells
        GPUArray<Dscalar2> cellPositions;
        //! VERTEX neighbors of every voronoi vertex
        GPUArray<int> vertexNeighbors;
        //! Cell neighbors of every voronoi vertex
        GPUArray<int> vertexCellNeighbors;

        //! Count the number of times "performTimeStep" has been called
        int Timestep;

        //!the box defining the periodic domain
        gpubox Box;

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!initialize the cuda RNG
        void initializeCurandStates(int gs, int i);

        //!Initialize AVM2D, set random orientations for vertex directors, prepare data structures
        void Initialize(int n,bool initGPU = true);

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(int n);

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        void computeGeometryGPU();

    protected:
        //!Number of cells in the simulation
        int Ncells;
        //!Number of vertices (i.e, degrees of freedom)
        int Nvertices;

        //!The time stepsize of the simulation
        Dscalar deltaT;

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
        //!An array of angles (relative to \hat{x}) that the vertex directors point
        GPUArray<Dscalar> vertexDirectors;

        //!The number of vertices defining each cell
        GPUArray<int> cellVertexNum;
        //!A structure that indexes the vertices defining each cell
        GPUArray<int> cellVertices;
        //!A 2dIndexer for computing where in the GPUArray to look for a given cell's vertices
        Index2D n_idx;
        //!An upper bound for the maximum number of neighbors that any cell has
        int vertexMax;
    };


#endif
