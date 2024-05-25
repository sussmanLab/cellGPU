#include "gradientDescent.h"
#include "utilities.cuh"
/*! \file gradientDescent.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
gradientDescent::gradientDescent(int _N, bool usegpu)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = usegpu;
    if(!GPUcompute)
        {
        displacements.neverGPU=true;
        sumReductions.neverGPU=true;
        sumReductionIntermediate.neverGPU=true;
        forceDotForce.neverGPU=true;
        };

    Ndof = _N;
    displacements.resize(Ndof);
    sumReductions.resize(1);
    sumReductionIntermediate.resize(Ndof);
    forceDotForce.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void gradientDescent::spatialSorting(const vector<int> &reIndexer)
    {
    //reIndexing = cellModel->returnItt();
    //reIndexRNG(noise.RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable
*/
void gradientDescent::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    cellModel = dynamic_pointer_cast<Simple2DCell>(model);
    }

/*!
Advances brownian dynamics by one time step
*/
void gradientDescent::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (cellModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = cellModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    };

/*!
The straightforward GPU implementation
*/
void gradientDescent::integrateEquationsOfMotionGPU()
    {
    cellModel->computeForces();
    gpu_copy_multipleOf_gpuarray(displacements,cellModel->returnForces(),deltaT);
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();
    };

double gradientDescent::calculateForceNorm()
    {
    double ans = 0;
    if(GPUcompute)
        {
        //force.force
            {
        ArrayHandle<double2> d_f(cellModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<double> d_ff(forceDotForce,access_location::device,access_mode::readwrite);
        gpu_dot_double2_vectors(d_f.data,d_f.data,d_ff.data,Ndof);
            }
        //sum reduction
            {
        ArrayHandle<double> d_ff(forceDotForce,access_location::device,access_mode::readwrite);
        ArrayHandle<double> d_intermediate(sumReductionIntermediate,access_location::device,access_mode::overwrite);
        ArrayHandle<double> d_assist(sumReductions,access_location::device,access_mode::overwrite);
        gpu_parallel_reduction(d_ff.data,d_intermediate.data,d_assist.data,0,Ndof);
            }
        ArrayHandle<double> h_assist(sumReductions,access_location::host,access_mode::read);
        ans = h_assist.data[0];
        }
    else
        {
        ArrayHandle<double2> h_f(cellModel->returnForces(),access_location::host,access_mode::read);
        for(int ii = 0; ii < Ndof; ++ii)
            {
            ans += h_f.data[ii].x*h_f.data[ii].x + h_f.data[ii].y*h_f.data[ii].y;
            }
        }
    return sqrt(ans);
    }

/*!
The straightforward CPU implementation
*/
void gradientDescent::integrateEquationsOfMotionCPU()
    {
    cellModel->computeForces();
    {//scope for array Handles
    ArrayHandle<double2> h_f(cellModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);

    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_disp.data[ii].x = deltaT*h_f.data[ii].x;
        h_disp.data[ii].y = deltaT*h_f.data[ii].y;
        };
    };//end array handle scope
    cellModel->moveDegreesOfFreedom(displacements);
    cellModel->enforceTopology();
    };
