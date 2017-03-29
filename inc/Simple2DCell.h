#ifndef SIMPLE2DCELL_H
#define SIMPLE2DCELL_H

#include "Simple2DModel.h"
#include "simpleEquationOfMotion.h"
#include "indexer.h"
#include "gpubox.h"
#include "HilbertSort.h"

/*! \file Simple2DCell.h */
//! Implement data structures and functions common to many off-lattice models of cells in 2D
/*!
A class defining some of the fundamental attributes and operations common to 2D off-lattice models
of cells. Note that while all 2D off-lattice models use some aspects of this base class, not all of
them are required to implement or use all of the below
*/
class Simple2DCell : public Simple2DModel
    {
    public:
        //!initialize member variables to some defaults
        Simple2DCell();

        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        virtual void setGPU(){GPUcompute = true;};

        //!Enforce CPU-only operation. Derived classes might have to do more work when the CPU mode is invoked
        virtual void setCPU(){GPUcompute = false;};

        //!get the number of degrees of freedom, defaulting to the number of cells
        virtual int getNumberOfDegreesOfFreedom(){return Ncells;};

        //!do everything necessary to compute forces in the current model
        virtual void computeForces(){};

        //!copy the models current set of forces to the variable
        virtual void getForces(GPUArray<Dscalar2> &forces){};

        //!default to returning forces on cells
        virtual GPUArray<Dscalar2> & returnForces(){return cellForces;};

        //!move the degrees of freedom
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> &displacements){};

        //!Do everything necessary to update or enforce the topology in the current model
        virtual void enforceTopology(){};

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set cell area and perimeter preferences according to input vector
        void setCellPreferences(vector<Dscalar2> &AreaPeriPreferences);

        //!Set random cell positions, and set the periodic box to a square with average cell area=1
        void setCellPositionsRandomly();

        //!Set cell positions according to a user-specified vector
        void setCellPositions(vector<Dscalar2> newCellPositions);
        //!Set vertex positions according to a user-specified vector
        void setVertexPositions(vector<Dscalar2> newVertexPositions);

        //! set uniform moduli for all cells
        void setModuliUniform(Dscalar newKA, Dscalar newKP);

        //!Set all cells to the same "type"
        void setCellTypeUniform(int i);
        //!Set cells to different "type"
        void setCellType(vector<int> &types);

        //!An uncomfortable function to allow the user to set vertex topology "by hand"
        void setVertexTopologyFromCells(vector< vector<int> > cellVertexIndices);

        //!return the base "itt" re-indexing vector
        virtual vector<int> & returnItt(){return itt;};

    //protected functions
    protected:
        //!set the size of the cell-sorting structures, initialize lists simply
        void initializeCellSorting();
        //!set the size of the vertex-sorting structures, initialize lists simply
        void initializeVertexSorting();
        //!Re-index cell arrays after a spatial sorting has occured.
        void reIndexCellArray(GPUArray<int> &array);
        //!why use templates when you can type more?
        void reIndexCellArray(GPUArray<Dscalar> &array);
        //!why use templates when you can type more?
        void reIndexCellArray(GPUArray<Dscalar2> &array);
        //!Re-index vertex after a spatial sorting has occured.
        void reIndexVertexArray(GPUArray<int> &array);
        //!why use templates when you can type more?
        void reIndexVertexArray(GPUArray<Dscalar> &array);
        //!why use templates when you can type more?
        void reIndexVertexArray(GPUArray<Dscalar2> &array);
        //!Perform a spatial sorting of the cells to try to maintain data locality
        void spatiallySortCells();
        //!Perform a spatial sorting of the vertices to try to maintain data locality
        void spatiallySortVertices();


    //public member variables
    public:
        //!Number of cells in the simulation
        int Ncells;
        //!Number of vertices
        int Nvertices;

        //! Cell positions... not used for computation, but can track, e.g., MSD of cell centers
        GPUArray<Dscalar2> cellPositions;
        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;
        //! VERTEX neighbors of every vertex
        /*!
        in general, we have:
        vertexNeighbors[3*i], vertexNeighbors[3*i+1], and vertexNeighbors[3*i+2] contain the indices
        of the three vertices that are connected to vertex i
        */
        GPUArray<int> vertexNeighbors;
        //! Cell neighbors of every vertex
        /*!
        in general, we have:
        vertexCellNeighbors[3*i], vertexCellNeighbors[3*i+1], and vertexCellNeighbors[3*i+2] contain
        the indices of the three cells are neighbors of vertex i
        */
        GPUArray<int> vertexCellNeighbors;
        //! CELL neighbors of every cell
        GPUArray<int> cellNeighbors;

        //!an array containing net force on each vertex
        GPUArray<Dscalar2> vertexForces;
        //!an array containing net force on each cell
        GPUArray<Dscalar2> cellForces;
        //!An array of integers labeling cell type...an easy way of determining if cells are different.
        /*!
        Please note that "type" is not meaningful unless it is used by child classes. That is, things
        like area/perimeter preferences, or motility, or whatever are neither set nor accessed by
        cell type, but rather by cell index! Thus, this is just an additional data structure that
        can be useful. For instance, the SPVTension2D classes uses the integers of cellType to
        determine when to apply an additional line tension between cells.
        */
        GPUArray<int> CellType;
        //!A indexer for turning a pair of cells into a 1-D index
        Index2D cellTypeIndexer;

    //protected member variables
    protected:
        //!the box defining the periodic domain
        gpubox Box;

        //!Compute aspects of the model on the GPU
        bool GPUcompute;

        //! A flag that determines whether the GPU RNG is the same every time.
        bool Reproducible;
        //!the area modulus
        Dscalar KA;
        //!The perimeter modulus
        Dscalar KP;
        //!The area and perimeter moduli of each cell. CURRENTLY NOT SUPPORTED, BUT EASY TO IMPLEMENT
        GPUArray<Dscalar2> Moduli;//(KA,KP)

        //!The current area and perimeter of each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell
        //!The area and perimeter preferences of each cell
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell
        //!The number of vertices defining each cell
        /*!
        cellVertexNum[c] is an integer storing the number of vertices that make up the boundary of cell c.
        */
        GPUArray<int> cellVertexNum;
        //!The number of CELL neighbors of each cell. For simple models this is the same as cellVertexNum, but does not have to be
        GPUArray<int> cellNeighborNum;
        //!A structure that indexes the vertices defining each cell
        /*!
        cellVertices is a large, 1D array containing the vertices associated with each cell.
        It must be accessed with the help of the Index2D structure n_idx.
        the index of the kth vertex of cell c (where the ordering is counter-clockwise starting
        with a random vertex) is given by
        cellVertices[n_idx(k,c)];
        */
        GPUArray<int> cellVertices;
        //!A 2dIndexer for computing where in the GPUArray to look for a given cell's vertices
        Index2D n_idx;
        //!An upper bound for the maximum number of neighbors that any cell has
        int vertexMax;
        //!3*Nvertices length array of the position of vertices around cells
        /*!
        For both AVM and SPV, it may help to save the relative position of the vertices around a
        cell, either for easy force computation or in the geometry routine, etc.
        voroCur.data[n_idx(nn,i)] gives the nth vertex, in CCW order, of cell i
        */
        GPUArray<Dscalar2> voroCur;
        //!3*Nvertices length array of the position of the last and next vertices along the cell
        //!Similarly, voroLastNext.data[n_idx(nn,i)] gives the previous and next vertex of the same
        GPUArray<Dscalar4> voroLastNext;

        //!A map between cell index and the spatially sorted version.
        /*!
        sortedArray[i] = unsortedArray[itt[i]] after a hilbert sort
        */
        vector<int> itt;
        //!A temporary structure that inverts itt
        vector<int> tti;
        //!To write consistent files...the cell that started the simulation as index i has current index tagToIdx[i]
        vector<int> tagToIdx;
        //!A temporary structure that inverse tagToIdx
        vector<int> idxToTag;
        //!A map between vertex index and the spatially sorted version.
        vector<int> ittVertex;
        //!A temporary structure that inverts itt
        vector<int> ttiVertex;
        //!To write consistent files...the vertex that started the simulation as index i has current index tagToIdx[i]
        vector<int> tagToIdxVertex;
        //!A temporary structure that inverse tagToIdx
        vector<int> idxToTagVertex;

        //utility data structures for interfacing with equations of motion
        //! a vector of Dscalars to be passed to the equation of motion
        vector<Dscalar> DscalarInfo;
        //! a vector of GPUArray of ints to be passed to the equation of motion
        vector<GPUArray<int> > IntArrayInfo;
        //! a vector of GPUArray of Dscalars to be passed to the equation of motion
        vector<GPUArray<Dscalar> > DscalarArrayInfo;
        //! a vector of GPUArray of Dscalar2s to be passed to the equation of motion
        vector<GPUArray<Dscalar2> > Dscalar2ArrayInfo;

        //!An array of displacements used only for the equations of motion
        GPUArray<Dscalar2> displacements;

    //reporting functions
    public:
        //!Get a copy of the particle positions
        void getPoints(GPUArray<Dscalar2> &ps){ps = cellPositions;};

        //!Report the current average force on each cell
        void reportMeanCellForce(bool verbose);
        //!Report the current average force per vertex...should be close to zero
        void reportMeanVertexForce()
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
        //! Report the average value of p/sqrt(A) for the cells in the system
        Dscalar reportq()
            {
            ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
            Dscalar A = 0.0;
            Dscalar P = 0.0;
            Dscalar q = 0.0;
            for (int i = 0; i < Ncells; ++i)
                {
                A = h_AP.data[i].x;
                P = h_AP.data[i].y;
                q += P / sqrt(A);
                };
            return q/(Dscalar)Ncells;
            };

        //temporary stuff for testing as I refactor
        virtual void setDeltaT(Dscalar dt) = 0;
        virtual void setCPU(bool a) = 0;
        virtual void setv0Dr(Dscalar a, Dscalar b) = 0;
    };
typedef shared_ptr<Simple2DCell> ForcePtr;
typedef weak_ptr<Simple2DCell> WeakForcePtr;

#endif
