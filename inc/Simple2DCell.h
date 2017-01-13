#ifndef SIMPLE2DCELL_H
#define SIMPLE2DCELL_H

#include "std_include.h"
#include "indexer.h"
#include "gpuarray.h"
#include "gpubox.h"
#include "curand.h"
#include "curand_kernel.h"

/*!
A class defining some of the fundamental attributes and operations common to 2D off-lattice models
of cells. This class will help refactor the AVM and SPV branches into a more coherent set. At the
moment the AVM2D class is based off of this, but DelaunayMD and SPV need to be refactored.
*/
//! Implement data structures and functions common to many off-lattice models of cells in 2D
class Simple2DCell
    {
    //public functions first
    public:
        //!Currently a vacant constructor
        Simple2DCell();

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set random cell positions, and set the periodic box to a square with average cell area=1
        void setCellPositionsRandomly();

        //!initialize the cuda RNG
        void initializeCurandStates(int gs, int i);

    //protected functions
    protected:
        //!Re-index arrays after a spatial sorting has occured.
        void reIndexArray(GPUArray<int> &array);
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar> &array);
        //!why use templates when you can type more?
        void reIndexArray(GPUArray<Dscalar2> &array);


    //public member variables
    public:
        //!Number of cells in the simulation
        int Ncells;
        //!Number of vertices (i.e, degrees of freedom)
        int Nvertices;

        //!the box defining the periodic domain
        gpubox Box;

        //!A flag that, when true, has performTimestep call the GPU routines
        bool GPUcompute;

        //! Count the number of times "performTimeStep" has been called
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;
        //! Cell positions... not used for computation, but can track, e.g., MSD of cell centers
        GPUArray<Dscalar2> cellPositions;
        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;
        /*!
        in general, we have:
        vertexNeighbors[3*i], vertexNeighbors[3*i+1], and vertexNeighbors[3*i+2] contain the indices
        of the three vertices that are connected to vertex i
        */
        //! CELL neighbors of every cell
        GPUArray<int> cellNeighbors;
        //! VERTEX neighbors of every vertex
        GPUArray<int> vertexNeighbors;
        /*!
        in general, we have:
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

        //!an array containing net force on each vertex
        GPUArray<Dscalar2> vertexForces;
        //!an array containing net force on each cell
        GPUArray<Dscalar2> cellForces;

        /*!
        For both AVM and SPV, it may help to save the relative position of the vertices around a
        cell, either for easy force computation or in the geometry routine, etc.
        */
        //!3*Nvertices length array of the position of vertices around cells
        GPUArray<Dscalar2> voroCur;
        //!3*Nvertices length array of the position of the last and next vertices along the cell
        GPUArray<Dscalar4> voroLastNext;

        /*!sortedArray[i] = unsortedArray[itt[i]] after a hilbert sort
        */
        //!A map between particle index and the spatially sorted version.
        vector<int> itt;
        //!A temporary structure that inverts itt
        vector<int> tti;
        //!To write consistent files...the particle that started the simulation as index i has current index tagToIdx[i]
        vector<int> tagToIdx;
        //!A temporary structure that inverse tagToIdx
        vector<int> idxToTag;



    //protected member variables
    protected:
        //!A flag to determine whether the CUDA RNGs should be initialized or not (so that the program will run on systems with no GPU by setting this to false
        bool initializeGPURNG;
        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> cellRNGs;
        //! A flag that determines whether the GPU RNG is the same every time.
        bool Reproducible;

        //!The current area and perimeter of each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell
        //!The area and perimeter preferences of each cell
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell
        /*!
        cellVertexNum[c] is an integer storing the number of vertices that make up the boundary of cell c.
        */
        //!The number of vertices defining each cell
        GPUArray<int> cellVertexNum;
        //!The number of CELL neighbors of each cell. For simple models this is the same as cellVertexNum, but does not have to be
        GPUArray<int> cellNeighborNum;
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

    //reporting functions
    public:
        //!Get a copy of the particle positions
        void getPoints(GPUArray<Dscalar2> &ps){ps = cellPositions;};

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
    };

#endif
