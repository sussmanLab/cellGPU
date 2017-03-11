#define ENABLE_CUDA

#include "selfPropelledCellVertexDynamics.h"
#include "selfPropelledParticleDynamics.cuh"
#include "selfPropelledCellVertexDynamics.cuh"
/*! \file selfPropelledCellVertexDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
selfPropelledCellVertexDynamics::selfPropelledCellVertexDynamics(int _Ncells, int _Nvertices)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = true;
    mu = 1.0;
    Ndof = _Nvertices;
    Nvertices = _Nvertices;
    Ncells = _Ncells;
    RNGs.resize(Ncells);
    };

/*!
\param globalSeed the global seed to use
\param offset the value of the offset that should be sent to the cuda RNG...
This is one part of what would be required to support reproducibly being able to load a state
from a databse and continue the dynamics in the same way every time. This is not currently supported.
*/
void selfPropelledCellVertexDynamics::initializeGPURNGs(int globalSeed, int offset)
    {
    if(RNGs.getNumElements() != Ncells)
        RNGs.resize(Ncells);
    ArrayHandle<curandState> d_curandRNGs(RNGs,access_location::device,access_mode::overwrite);
    int globalseed = globalSeed;
    if(!Reproducible)
        {
        clock_t t1=clock();
        globalseed = (int)t1 % 100000;
        printf("initializing curand RNG with seed %i\n",globalseed);
        };
    gpu_initialize_RNG(d_curandRNGs.data,Ncells,offset,globalseed);
    };

/*!
Given a vector of forces, update the displacements array to the amounts needed to advance the simulation one step
For the self-propelled particle model, the data buckest need to be the following:
DscalarInfo = {} (empty vector)
DscalarArrayInfo[0] = cellDirectors
Dscalar2ArrayInfo[0] = forces
Dscalar2ArrayInfo[1] = Motility (each entry is (v0 and Dr) per cell)
IntArrayInfo = {} (empty vector)
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotion(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo,
        vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> > &IntArrayInfo, GPUArray<Dscalar2> &displacements)
    {
    Timestep += 1;
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU(DscalarInfo,DscalarArrayInfo,Dscalar2ArrayInfo,IntArrayInfo,displacements);
        }
    else
        {
        integrateEquationsOfMotionCPU(DscalarInfo,DscalarArrayInfo,Dscalar2ArrayInfo, IntArrayInfo, displacements);
        }
    };



/*!
The straightforward GPU implementation
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotionGPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo,
        vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> > &IntArrayInfo, GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> d_f(Dscalar2ArrayInfo[0],access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_cd(DscalarArrayInfo[0],access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_motility(Dscalar2ArrayInfo[1],access_location::device,access_mode::read);
    ArrayHandle<int> d_vcn(IntArrayInfo[0],access_location::device,access_mode::read);

    ArrayHandle<curandState> d_RNG(RNGs,access_location::device,access_mode::readwrite);

    gpu_spp_cellVertex_eom_integration(d_f.data,
                 d_disp.data,
                 d_motility.data,
                 d_cd.data,
                 d_vcn.data,
                 d_RNG.data,
                 Nvertices,Ncells,
                 deltaT,
                 Timestep,
                 mu);
    };

/*!
Move every vertex according to the net force on it and its motility...CPU routine
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotionCPU(vector<Dscalar> &DscalarInfo, vector<GPUArray<Dscalar> > &DscalarArrayInfo,
        vector<GPUArray<Dscalar2> > &Dscalar2ArrayInfo, vector<GPUArray<int> > &IntArrayInfo, GPUArray<Dscalar2> &displacements)
    {
    ArrayHandle<Dscalar2> h_f(Dscalar2ArrayInfo[0],access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(DscalarArrayInfo[0],access_location::host,access_mode::readwrite);
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_motility(Dscalar2ArrayInfo[1],access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(IntArrayInfo[0],access_location::host,access_mode::read);

    normal_distribution<> normal(0.0,1.0);

    Dscalar directorx,directory;
    Dscalar2 disp;
    for (int i = 0; i < Nvertices; ++i)
        {
        Dscalar v1 = h_motility.data[h_vcn.data[3*i]].x;
        Dscalar v2 = h_motility.data[h_vcn.data[3*i+1]].x;
        Dscalar v3 = h_motility.data[h_vcn.data[3*i+2]].x;
        //for uniform v0, the vertex director is the straight average of the directors of the cell neighbors
        directorx  = v1*cos(h_cd.data[ h_vcn.data[3*i] ]);
        directorx += v2*cos(h_cd.data[ h_vcn.data[3*i+1] ]);
        directorx += v3*cos(h_cd.data[ h_vcn.data[3*i+2] ]);
        directorx /= 3.0;
        directory  = v1*sin(h_cd.data[ h_vcn.data[3*i] ]);
        directory += v2*sin(h_cd.data[ h_vcn.data[3*i+1] ]);
        directory += v3*sin(h_cd.data[ h_vcn.data[3*i+2] ]);
        directory /= 3.0;
        //move vertices
        h_disp.data[i].x = deltaT*(directorx + mu*h_f.data[i].x);
        h_disp.data[i].y = deltaT*(directory + mu*h_f.data[i].y);
        };

    //update cell directors
    for (int i = 0; i < Ncells; ++i)
        {
        Dscalar randomNumber;
        if (Reproducible)
            randomNumber = normal(gen);
        else
            randomNumber = normal(genrd);
        Dscalar Dr = h_motility.data[i].y;
        h_cd.data[i] += randomNumber*sqrt(2.0*deltaT*Dr);
        };
    };
