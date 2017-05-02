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
//    cout <<mat << endl;
//    cout << es.eigenvalues() << endl;
    };
