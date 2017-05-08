#ifndef EigenMatrix_H
#define EigenMatrix_H

#include "std_include.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

/*! \file spv2d.h */

//!Implements an interface to an Eigen dense matrix, and allows for an eigendecomposition to be done
class EigMat
    {
    public:
        EigMat(){};
        EigMat(int n);
        //!The internal dense matrix
        Eigen::MatrixXd mat;
        //!A vector of eigenvalues
        vector<Dscalar> eigenvalues;
        //!A vector of vector of eigenvectors
        vector< vector< Dscalar> > eigenvectors;
        //!Set the internal matrix to a square matrix of all zero elements
        void setMatrixToZero(int n);
        //! set M_{ij}= M_{ji}=value
        void placeElementSymmetric(int row, int col, Dscalar value);
        //! use the self-adjoint eigensolver, and save the eigenvalues
        void SASolve(int vectors = 0);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    };


#endif

