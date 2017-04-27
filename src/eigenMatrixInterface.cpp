#include "eigenMatrixInterface.h"

/*! \file eigenMatrixInterface.cpp */

EigMat::EigMat(int n)
    {
    setMatrixToZero(n);
    };

void EigMat::setMatrixToZero(int n)
    {
    mat = Eigen::MatrixXd::Zero(n, n);
    };

void EigMat::placeElementSymmetric(int row, int col, Dscalar value)
    {
    mat(row,col) = value;
    mat(col,row) = value;
    };

void EigMat::SASolve()
    {
    int size = mat.rows();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(mat);

    eigenvalues.resize(size);
    for (int ee = 0; ee < size; ++ee)
        eigenvalues[ee]=es.eigenvalues()[ee];

//    cout <<mat << endl;
//    cout << es.eigenvalues() << endl;
    };
