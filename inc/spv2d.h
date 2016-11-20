//spv.h
#ifndef SPV_H
#define SPV_H

using namespace std;

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
        float deltaT;
        float Dr;
        float v0;

        float gamma; //value of inter-cell surface tension
        bool useTension;

        GPUArray<float2> VoronoiPoints;
        GPUArray<float2> AreaPeriPreferences;
        GPUArray<float2> AreaPeri;
        GPUArray<float2> Moduli;//(KA,KP)
        GPUArray<int> CellType;//(KA,KP)

        GPUArray<float> cellDirectors;
        GPUArray<float2> displacements;

        int Timestep;
        curandState *devStates;

        //delSet.data[n_idx(nn,i)] are four consecutive delaunay neighbors, orientationally ordered, of point i (for use in computing forces on GPU)
        GPUArray<int4> delSets;
        //delOther.daata[n_idx(nn,i)] contains the index of the "other" delaunay neighbor. i.e., the mutual neighbor of delSet.data[n_idx(nn,i)].y and delSet.data[n_idx(nn,i)].z that isn't point i
        GPUArray<int> delOther;
        GPUArray<float2> forceSets;

    public:
        GPUArray<float2> forces;

        ~SPV2D()
            {
            cudaFree(devStates);
            };
        //initialize with random positions in a square box
        SPV2D(int n);
        //additionally set all cells to have uniform target A_0 and P_0 parameters
        SPV2D(int n, float A0, float P0);

        //initialize DelaunayMD, and set random orientations for cell directors
        void Initialize(int n);

        //set and get
        void setDeltaT(float dt){deltaT = dt;};
        void setv0(float v0new){v0 = v0new;};
        void setDr(float dr){Dr = dr;};

        void setTension(float g){gamma = g;};
        void setUseTension(bool u){useTension = u;};


        void setCellPreferencesUniform(float A0, float P0);
        void setCellTypeUniform(int i);
        void setCellType(vector<int> &types);
        void setModuliUniform(float KA, float KP);

        void setCurandStates(int i);

        //utility
        void getDelSets(int i);
        void allDelSets();
        void centerCells();

        //cell-dynamics related functions
        void performTimestep();
        void performTimestepCPU();
        void performTimestepGPU();


        //CPU functions
        void computeGeometryCPU();
        void computeSPVForceCPU(int i);
        void computeSPVForceWithTensionsCPU(int i,bool verbose = false);
        void calculateDispCPU();


        //GPU functions
        void DisplacePointsAndRotate();
        void computeGeometry();
        void computeSPVForcesGPU();
        void computeSPVForcesWithTensionsGPU();



        //


        //testing functions...
        void reportForces();
        void meanForce();
        void meanArea();
        float reportq();

        float triangletiming, forcetiming;
    };





#endif
