#ifndef ENERGYMINIMIZERNEWTONRAPHSON_H
#define ENERGYMINIMIZERNEWTONRAPHSON_H

#include "std_include.h"
#include "functions.h"
#include "gpuarray.h"
#include "Simple2DCell.h"
#include "eigenMatrixInterface.h"

/*! \file EnergyMinimizerNewtonRaphson.h */
//!Implement energy minimization via the Newton-Raphson algorithm
/*!
*/
class EnergyMinimizerNewtonRaphson : public simpleEquationOfMotion
    {
    public:
        //!The basic constructor
        EnergyMinimizerNewtonRaphson(){};
        //!The basic constructor that feeds in a target system to minimize
        EnergyMinimizerNewtonRaphson(shared_ptr<Simple2DModel> system);
        //!Set a bunch of default initialization parameters (if the State is available to determine the size of vectors)
        void initializeFromModel();

        //!The system that can compute forces, move degrees of freedom, etc.
        shared_ptr<Simple2DModel> State;
        //!set the internal State to the given model
        virtual void set2DModel(shared_ptr<Simple2DModel> &_model){State = _model;};

        //!Set the maximum number of iterations before terminating (or set to -1 to ignore)
        void setMaximumIterations(int maxIt){maxIterations = maxIt;};
        //!Set the force cutoff
        void setForceCutoff(Dscalar fc){forceCutoff = fc;};
        //!Enforce GPU-only operation. This is the default mode, so this method need not be called most of the time.
        void setGPU(){GPUcompute = true;};
        //!Enforce CPU-only operation.
        virtual void setCPU(){GPUcompute = false;};

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();
        //!The "intergate equatios of motion just calls minimize
        virtual void integrateEquationsOfMotion(){minimize();};

        //!Use the force function of State to compute the gradient
        void getGradient();
        //!Use the Eigen interface to compute the Hessian
        void getHessian();


    protected:
        //!The eigenmatrix interface
        EigMat EM;
        //! The eigen gradient
        Eigen::VectorXd eGradient;
        //! The eigen displacement
        Eigen::VectorXd eDisplace;
        //! The eigen inverse hessian
        Eigen::MatrixXd Hinverse;
        //!The number of iterations performed
        int iterations;
        //!The maximum number of iterations allowed
        int maxIterations;
        //!The cutoff value of the maximum force
        Dscalar forceMax;
        //!The cutoff value of the maximum force
        Dscalar forceCutoff;
        //!The number of points, or cells, or particles
        int N;
        //!The GPUArray containing the force
        GPUArray<Dscalar2> force;
        //!an array of displacements
        GPUArray<Dscalar2> displacement;

        //!Should calculations be done on the GPU?
        bool GPUcompute;

        //!The value added to the diagonal to stabilize the inverse
        Dscalar tether;
    };


#endif
