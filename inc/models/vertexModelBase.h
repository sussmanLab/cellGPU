#ifndef vertexModelBase_H
#define vertexModelBase_H

#include "simpleVertexModelBase.h"
#include "Simulation.h"

/*! \file vertexModelBase.h */
//!A class that can calculate many geometric and topological features common to three-fold coordinated vertex models
/*!
This class focuses on many energy-functional independent featuers of 2D vertex models that are further
specialized by having all vertices by three-fold coordinated. It can compute geometric features of cells,
such as their area and perimeter; and it is where a geometric form of cell division and death is implemented.

Vertex models have to maintain the topology of the cell network by hand, so child classes need to
implement not only an energy functional (and corresponding force law), but also rules for topological
transitions. This base class will implement a very simple T1 transition scheme
Only T1 transitions are currently implemented, and they occur whenever two vertices come closer
than a set threshold distance.
 */

class vertexModelBase : public simpleVertexModelBase
    {
    public:
        //!Initialize vertexModelBase, set random orientations for vertex directors, prepare data structures
        void initializeVertexModelBase(int n);

        //!Initialize cells to be a voronoi tesselation of a random point set
        void setCellsVoronoiTesselation();

        //!Compute the geometry (area & perimeter) of the cells on the CPU
        virtual void computeGeometryCPU();
        //!Compute the geometry (area & perimeter) of the cells on the GPU
        virtual void computeGeometryGPU();

        //!Divide cell...vector should be cell index i, vertex 1 and vertex 2
        virtual void cellDivision(const vector<int> &parameters,const vector<Dscalar> &dParams = {});

        //!Kill the indexed cell...cell must have only three associated vertices
        virtual void cellDeath(int cellIndex);

        //!Simple test for T1 transitions (edge length less than threshold) on the CPU
        void testAndPerformT1TransitionsCPU();
        //!Simple test for T1 transitions (edge length less than threshold) on the GPU...calls the following functions
        void testAndPerformT1TransitionsGPU();

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

    protected:
        //!Initialize the data structures for edge flipping...should also be called if Nvertices changes
        void initializeEdgeFlipLists();
        //! data structure to help with cell-vertex list
        GPUArray<int> growCellVertexListAssist;

        //!test the edges for a T1 event, and grow the cell-vertex list if necessary
        void testEdgesForT1GPU();
        //!perform the edge flips found in the previous step
        void flipEdgesGPU();

        //!For finding T1s on the CPU; find the set of vertices and cells involved in the transition
        void getCellVertexSetForT1(int v1, int v2, int4 &cellSet, int4 &vertexSet, bool &growList);

        //! data structure to help with not simultaneously trying to flip nearby edges
        GPUArray<int> finishedFlippingEdges;
    };
#endif
