//spv.h
#ifndef SPV_H
#define SPV_H

using namespace std;

#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"

#include "Matrix.h"   
#include "cu_functions.h"

#include "DelaunayMD.h"

class SPV2D : public DelaunayMD
    {
    protected:
//        GPUArray<float2> points;      //vector of particle positions
        float deltaT;
        float Dr;

        GPUArray<float2> AreaPeriPreferences;
        GPUArray<float2> AreaPeri;
        
        GPUArray<float2> cellDirectors;
        GPUArray<float2> forces;

    public:
        //initialize with random positions in a square box
        SPV2D(int n);
        //additionally set all cells to have uniform target A_0 and P_0 parameters
        SPV2D(int n, float A0, float P0);

        //initialize DelaunayMD, and set random orientations for cell directors
        void Initialize(int n);

        //set and get
        void setDeltaT(float dt){deltaT = dt;};
        void setDr(float dr){Dr = Dr;};
        void setCellPreferencesUniform(float A0, float P0);

        //cell-dynamics related functions
        void computeSPVForces();
        void performTimestep();

        void computeSPVForceCPU(int i);
        void computeGeometryCPU();


        //testing functions...
        void meanForce();
        void meanArea();
    };





#endif
