//spv.h
#ifndef SPV_H
#define SPV_H

using namespace std;

#include <stdio.h>
#include <cmath>
#include <random>
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

        GPUArray<float2> VoronoiPoints;
        GPUArray<float2> AreaPeriPreferences;
        GPUArray<float2> AreaPeri;
        GPUArray<float2> Moduli;//(KA,KP)
        GPUArray<int> CellType;//(KA,KP)
        
        GPUArray<float> cellDirectors;
        GPUArray<float2> displacements;

        int Timestep;
        curandState *devStates;

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
        void setCellPreferencesUniform(float A0, float P0);
        void setCellTypeUniform(int i);
        void setCellType(vector<int> &types);
        void setModuliUniform(float KA, float KP);

        void setCurandStates(int i);

        //cell-dynamics related functions
        void performTimestep();
        void performTimestepCPU();
        void performTimestepGPU();

        void computeGeometry();

        void computeGeometryCPU();
        void computeSPVForceCPU(int i);
        void computeSPVForceWithTensionsCPU(int i,float Gamma,bool verbose = false);
        void calculateDispCPU();


        void DisplacePointsAndRotate();
        //still need to write
        //
        ////for compute geo GPU...add new voro structure that matches the neighbor list, save voro points?
        void computeGeometryGPU();

        void computeSPVForcesGPU();


        //


        //testing functions...
        void reportForces();
        void meanForce();
        void meanArea();
        float reportq();

        float triangletiming, forcetiming;
    };





#endif
