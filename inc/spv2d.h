//spv.h
#ifndef SPV_H
#define SPV_H


#include "std_include.h"
#include <stdio.h>
#include <cmath>
#include <random>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "vector_types.h"
#include "vector_functions.h"

#include "Matrix.h"
#include "cu_functions.h"

#include "DelaunayMD.h"

class SPV2D : public DelaunayMD
    {
    protected:

        //are inter-cell tensions to be added? are there any force exclusions?
        bool useTension;
        bool particleExclusions;

        //for mono-motile systems, the value of Dr and v_0. Just used for reporting purposes
        Dscalar Dr;
        Dscalar v0;

        //value of inter-cell surface tension
        Dscalar gamma;

        //arrays of cell material, motility, and preference parameters.
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell
        GPUArray<Dscalar2> Moduli;//(KA,KP)
        GPUArray<Dscalar2> Motility;//(v0,Dr) for each cell

        //a vector of displacements used in the CPU branch but not the GPU branch on each timestep
        GPUArray<Dscalar2> displacements;

        //a vector of random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> devStates;

        //delSet.data[n_idx(nn,i)] are the previous and next consecutive delaunay neighbors, orientationally ordered, of point i (for use in computing forces on GPU)
        GPUArray<int2> delSets;
        //delOther.data[n_idx(nn,i)] contains the index of the "other" delaunay neighbor. i.e., the mutual neighbor of delSet.data[n_idx(nn,i)].y and delSet.data[n_idx(nn,i)].z that isn't point i
        GPUArray<int> delOther;

        //similarly, VoroCur.data[n_idx(nn,i)] gives the nth voronoi vertex, in order, of particle i
        //VoroLastNext.data[n_idx(nn,i)] gives the previous and next voronoi vertex of the same
        GPUArray<Dscalar2> VoroCur;
        GPUArray<Dscalar4> VoroLastNext;

        //in GPU mode, interactions are computed "per voronoi vertex"...forceSets are summed up to get total force on a particle
        GPUArray<Dscalar2> forceSets;

    public:
        //how many times has "performTimeStep" been called?
        int Timestep;
        //how frequently should the spatial sorter be used?
        int sortPeriod;
        bool spatialSortThisStep;

        //what is the timestep size of the simulation?
        Dscalar deltaT;
        //an array of integers labeling cell type
        GPUArray<int> CellType;
        //an array of angles (relative to \hat{x}) that the cell directors point
        GPUArray<Dscalar> cellDirectors;
        //an array of forces on cels
        GPUArray<Dscalar2> forces;

        //"exclusions" zero out the force on a cell...the external force needed to do this is stored in external_forces
        GPUArray<Dscalar2> external_forces;
        GPUArray<int> exclusions;

        //constructors and destructors
        //initialize with random positions in a square box
        SPV2D(int n);
        //additionally set all cells to have uniform target A_0 and P_0 parameters
        SPV2D(int n, Dscalar A0, Dscalar P0);
        //previous iterations of the code needed an explicit destructor for some
        //"cudaFree" calls
        //~SPV2D()
        //    {
        //    };


        //initialize DelaunayMD, and set random orientations for cell directors
        void Initialize(int n);


        //set and get
        void setDeltaT(Dscalar dt){deltaT = dt;};
        void setv0Dr(Dscalar v0new,Dscalar drnew);
        void setSortPeriod(int sp){sortPeriod = sp;};
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);
        void setCellTypeUniform(int i);
        void setCellType(vector<int> &types);
        void setModuliUniform(Dscalar KA, Dscalar KP);
        void setTension(Dscalar g){gamma = g;};
        void setUseTension(bool u){useTension = u;};
        void setCellMotility(vector<Dscalar> &v0s,vector<Dscalar> &drs);

        //specialty functions for setting cell types in particular geometries
        //sets particles within an ellipse to type 0, outside to type 1. frac is fraction of area for the ellipse to take up, aspectRatio is (r_x/r_y)
        void setCellTypeEllipse(Dscalar frac, Dscalar aspectRatio);
        //sets particles within a strip (surface normal to x) to type 0, other particles to type 1. Fraction is the area of strip occupied by the system
        void setCellTypeStrip(Dscalar frac);


        //set exclusions...if a particle is excluded (ex[idx]=1) then its force is zeroed out (the external force to do this is stored in excluded_forces) and its motility is set to zero
        void setExclusions(vector<int> &exes);

        //internal utilities...
        //maintain topological data structures
        void getDelSets(int i);
        void allDelSets();
        //resize neighMax-related lists
        void resetLists();
        //initialize the cuda RNG
        void setCurandStates(int i);
        //sort points along a Hilbert curve for data locality
        void spatialSorting();

        //cell-dynamics related functions...these call functions in the next section
        //in general, these functions are the common calls, and test flags to know whether to call specific versions of specialty functins
        void performTimestep();
        void performTimestepCPU();
        void performTimestepGPU();
        void ComputeForceSetsGPU();
        void SumForcesGPU();


        //CPU functions
        void computeGeometryCPU();
        void computeSPVForceCPU(int i);
        void computeSPVForceWithTensionsCPU(int i,bool verbose = false);
        void calculateDispCPU();


        //GPU functions
        void DisplacePointsAndRotate();
        void computeGeometryGPU();
        void computeSPVForceSetsGPU();
        void computeSPVForceSetsWithTensionsGPU();
        void sumForceSets();
        void sumForceSetsWithExclusions();


        //testing and reporting functions...
        void reportCellInfo();
        void reportForces();
        void meanForce();
        void meanArea();
        Dscalar reportq();

        Dscalar triangletiming, forcetiming;
    };





#endif
