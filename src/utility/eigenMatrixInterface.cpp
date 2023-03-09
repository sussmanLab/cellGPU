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

void EigMat::placeElement(int row, int col, double value)
    {
    mat(row,col) = value;
    };

void EigMat::placeElementSymmetric(int row, int col, double value)
    {
    placeElement(row,col,value);
    placeElement(col,row,value);
    };

void EigMat::SASolve(int vectors)
    {
    int size = mat.rows();
    es.compute(mat);

    eigenvalues.resize(size);
    for (int ee = 0; ee < size; ++ee)
        eigenvalues[ee]=es.eigenvalues()[ee];

    if (vectors > 0)
        {
        eigenvectors.resize(vectors);
        for (int vec = 0; vec < vectors; ++vec)
            {
            eigenvectors[vec].resize(size);
            for (int vv = 0; vv < size; ++vv)
                eigenvectors[vec][vv] = es.eigenvectors().col(vec)[vv];
            };
        };
    };

void EigMat::getEvec(int i, vector<double> &vec)
    {
    vec.resize(mat.rows());
    for (int j = 0; j < mat.rows(); ++j)
        {
        if (i < eigenvectors.size())
            vec[j] = eigenvectors[i][j];
        else
            vec[j] = es.eigenvectors().col(i)[j];
        };
    };

