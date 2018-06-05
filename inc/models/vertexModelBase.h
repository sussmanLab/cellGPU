#ifndef vertexModelBase_H
#define vertexModelBase_H

#include "Simple2DActiveCell.h"
//include spp dynamics for SPV-based initialization of configurations
#include "selfPropelledParticleDynamics.h"
#include "Simulation.h"

/*! \file vertexModelBase.h */
//!A class that can calculate many geometric and topological features common to vertex models
/*!
This class captures many energy-functional-independent features common to 2D vertex models of cells.
It can compute geometric features of cells, such as their area and perimeter; it knows that in vertex
models the degrees of freedom are the vertices (for returning the right vector of forces, moving
d.o.f. around, etc.); and it is where a geometric form of cell division is implemented.

Vertex models have to maintain the topology of the cell network by hand, so child classes need to
implement not only an energy functional (and corresponding force law), but also rules for topological
transitions. This base class will implement a very simple T1 transition scheme
Only T1 transitions are currently implemented, and they occur whenever two vertices come closer
than a set threshold distance. All vertices are three-valent.
 */

class vertexModelBase : public Simple2DActiveCell
    {
    public:
        //!In vertex models the number of degrees of freedom is the number of vertices
        virtual int getNumberOfDegreesOfFreedom(){return Nvertices;};

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<Dscalar2> & displacements,Dscalar scale = 1.);

        //!return the forces
        virtual void getForces(GPUArray<Dscalar2> &forces){forces = vertexForces;};

        //!Initialize vertexModelBase, set random orientations for vertex directors, prepare data structures
        void initializeVertexModelBase(int n,bool spvInitialize = false);

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation(bool spvInitialize = false);

        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<Dscalar2> & returnForces(){return vertexForces;};
        //!return a reference to the GPUArray of the current velocities
        virtual GPUArray<Dscalar2> & returnVelocities(){return vertexVelocities;};
        //!return a reference to the GPUArray of the current positions
        virtual GPUArray<Dscalar2> & returnPositions(){return vertexPositions;};
        //!return a reference to the GPUArray of the current masses
        virtual GPUArray<Dscalar> & returnMasses(){return vertexMasses;};

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        virtual void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        virtual void computeGeometryGPU();

        //!Call the CPU or GPU getCellCentroids function
        void getCellCentroids();
        //!Get the cell position from the vertices on the CPU
        void getCellCentroidsCPU();
        //!Get the cell position from the vertices on the GPU
        void getCellCentroidsGPU();
        //!Call the CPU or GPU getCellPositions function
        void getCellPositions();
        //!Get the cell position from the average vertex position on the CPU
        void getCellPositionsCPU();
        //!Get the cell position from the average vertex position on the GPU
        void getCellPositionsGPU();

        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(const vector<int> &parameters,const vector<Dscalar> &dParams = {});

        //!Kill the indexed cell...cell must have only three associated vertices
        virtual void cellDeath(int cellIndex);

        //!Set the length threshold for T1 transitions
        virtual void setT1Threshold(Dscalar t1t){T1Threshold = t1t;};

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

        //!spatially sort the *vertices* along a Hilbert curve for data locality
        virtual void spatialSorting();

        //!update/enforce the topology, performing simple T1 transitions
        virtual void enforceTopology();

        /*!
        if vertexEdgeFlips[3*i+j]=1 (where j runs from 0 to 2), the the edge connecting vertex i and vertex
        vertexNeighbors[3*i+j] has been marked for a T1 transition
        */
        //! flags that indicate whether an edge should be GPU-flipped (1) or not (0)
        GPUArray<int> vertexEdgeFlips;
        //! it is important to not flip edges concurrently, so this data structure helps flip edges sequentially
        GPUArray<int> vertexEdgeFlipsCurrent;

        /*!
        vertexForceSets[3*i], vertexForceSets[3*i+1], and vertexForceSets[3*i+2] contain the contribution
        to the net force on vertex i due to the three cell neighbors of vertex i
        */
        //!an array containing the three contributions to the force on each vertex
        GPUArray<Dscalar2> vertexForceSets;

        //!A threshold defining the edge length below which a T1 transition will occur
        Dscalar T1Threshold;

        //!Enforce CPU-only operation.
        void setCPU(bool global = true){GPUcompute = false;};

    protected:
        //!if the maximum number of vertices per cell increases, grow the cellVertices list
        void growCellVerticesList(int newVertexMax);

        //!Initialize the data structures for edge flipping...should also be called if Nvertices changes
        void initializeEdgeFlipLists();
        //! data structure to help with cell-vertex list
        GPUArray<int> growCellVertexListAssist;

        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();

        //utility functions
        //!For finding T1s on the CPU; find the set of vertices and cells involved in the transition
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

        //! data structure to help with not simultaneously trying to flip nearby edges
        GPUArray<int> finishedFlippingEdges;

        //! data structure per cell for not simulataneously flipping nearby edges
        GPUArray<int> cellEdgeFlips;
        //! data structure per cell for not simulataneously flipping nearby edges
        GPUArray<int4> cellSets;
    //reporting functions
    public:
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
    };

#endif
