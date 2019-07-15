#define ENABLE_CUDA

#include "vertexModelGeneric.h"
/*! \file vertexModelGeneric.cpp */

/*!
Take care of all base class initialization functions, this involves setting arrays to the right size, etc.
*/
void vertexModelGeneric::enforceTopology()
    {
    if(GPUcompute)
        {
        enforceTopologyGPU();
        return;
        };

    /*
    This is where one needs to decide rules for how to *use* the topological routines that are
    permitted in the generic vertex model.
    Here are the allowable functions:

    //T1-like functions
    performT1Transition(int vertexIndex1, int vertexIndex2) -- take two vertices (that should be connected!), and force a T1 transtion. No restriction on coordination of vertices. After the T1 at the moment the new edge length is a small multiple of the T1Threshold
    mergeVertices(vector<int> verticesToMerge) -- merge any number of vertices into a single vertex; the new vertex has all of the (cell and vertex) connectivity of all of the former vertices
    splitVertex(int vertexIndex, Dscalar separation, Dscalar theta) -- Take a vertex and divide it into two vertices, separated by some distance at some angle (lab frame)
    subdivideEdge(int vertexIndex1, int vertexIndex2) -- Take an edge (specified by it's two vertices) and add a new vertex in the middle

    //T2-like functions
    cellDivision(const vector<int> &parameters,const vector<Dscalar> &dParams) -- the first vector of ints (which is all you need) specifice the cell index and two numbers that should be less than or equal to the number of vertices. The cell divides, with two new vertices and a new dividing edge created (where the two vertex numbers specify in CCW order which edges get to be the dividing edges)
    cellDeath(int cellIndex) -- kills a cell, and merges its former vertices to a single vertex
    removeCells(vector<int> cellIndices) -- "Remove" cells whose index matches those in the vector...This function will delete a cell but leave its vertices (as long as the vertex is part of at least one cell...useful for creating open boundaries)

    //T3-like functions
    (Nothing at the moment -- in principle one can do this by checking if the vertex coordination of a
     vertex is greater than the number of cells it is adjacent to. Not hard, but I don't have a reason
     to implement this at the moment).
    */


    };

void vertexModelGeneric::enforceTopologyGPU()
    {
    UNWRITTENCODE("GPU ROUTINE NOT WRITTEN");
    }
