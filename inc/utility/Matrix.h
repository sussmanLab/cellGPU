#ifndef Matrix_H
#define Matrix_H

#include "std_include.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file Matrix.h */
//!contains a {{x11,x12},{x21,x22}} set, and matrix manipulations
/*!
Matrix2x2 provides a simple interface for operations using 2x2 matrices. In particular, it implement
matrix-maxtrix multiplication, and has specialized matrix-vector and vector-matrix multiplication in
which Dscalar2 variables take the place of vectors. A dyadic product is implemented which takes two
Dscalar2s and returns a Matrix2x2
*/
struct Matrix2x2
    {
    public:
        //!The entries of the matrix
        Dscalar x11, x12, x21, x22;
        //!Default constructor is the identity matrix
        HOSTDEVICE Matrix2x2() : x11(1.0), x12(0.0), x21(0.0),x22(1.0) {};
        //!Generic constructor is whatever you wnat it to be
        HOSTDEVICE Matrix2x2(Dscalar y11, Dscalar y12, Dscalar y21,Dscalar y22) : x11(y11), x12(y12), x21(y21),x22(y22) {};

        //!Set the values to some desired set
        HOSTDEVICE void set(Dscalar y11, Dscalar y12, Dscalar y21, Dscalar y22)
                            {
                            x11=y11; x12=y12;x21=y21;x22=y22;
                            };

        //!Transpose
        HOSTDEVICE void transpose()
                            {
                            Dscalar y21,y12;
                            y21=x12;
                            y12=x21;
                            x12=y12;
                            x21=y21;
                            };

        //!assignment operator
        HOSTDEVICE void operator=(const Matrix2x2 &m2)
                            {
                            set(m2.x11,m2.x12,m2.x21,m2.x22);
                            };
        //!matrix multiplication operator
        HOSTDEVICE void operator*=(const Matrix2x2 &m2)
                            {
                            set(x11*m2.x11 + x12*m2.x21,
                                x11*m2.x12 + x12*m2.x22,
                                x21*m2.x11 + x22*m2.x21,
                                x21*m2.x12 + x22*m2.x22
                                );
                            };

        //!matrix multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp*=m2;
                            return temp;
                            };

        //!scalar multiplication operator
        HOSTDEVICE void operator*=(Dscalar a)
                            {
                            set(a*x11,a*x12,a*x21,a*x22);
                            };

        //!scalar right multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const Matrix2x2 &m,const Dscalar a)
                            {
                            Matrix2x2 temp(m);
                            temp*=a;
                            return temp;
                            };

        //!scalar left multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const Dscalar a, const Matrix2x2 &m)
                            {
                            Matrix2x2 temp(m);
                            temp*=a;
                            return temp;
                            };

        //!Matrix addition operator
        HOSTDEVICE void operator+=(const Matrix2x2 &m2)
                            {
                            set(x11+m2.x11,
                                x12+m2.x12,
                                x21+m2.x21,
                                x22+m2.x22
                               );
                            };

        //!Matrix addition operator
        HOSTDEVICE friend Matrix2x2 operator+(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp+=m2;
                            return temp;
                            };

        //!Matrix subtraction operator
        HOSTDEVICE void operator-=(const Matrix2x2 &m2)
                            {
                            set(x11-m2.x11,
                                x12-m2.x12,
                                x21-m2.x21,
                                x22-m2.x22
                               );
                            };

        //!matrix subtraction operator
        HOSTDEVICE friend Matrix2x2 operator-(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp-=m2;
                            return temp;
                            };

        //!matrix-vector multiplication operator
        HOSTDEVICE friend Dscalar2 operator*(const Dscalar2 &v, const Matrix2x2 &m)
                            {
                            Dscalar2 temp;
                            temp.x = v.x*m.x11 + v.y*m.x21;
                            temp.y = v.x*m.x12 + v.y*m.x22;
                            return temp;
                            };

        //!matrix-vector multiplication operator
        HOSTDEVICE friend Dscalar2 operator*(const Matrix2x2 &m, const Dscalar2 &v)
                            {
                            Dscalar2 temp;
                            temp.x = m.x11*v.x+m.x12*v.y;
                            temp.y = m.x21*v.x+m.x22*v.y;
                            return temp;
                            };

    };

//! Form a matrix by the dyadic product of two vectors
HOSTDEVICE Matrix2x2 dyad(const Dscalar2 &v1, const Dscalar2 &v2)
    {
    return Matrix2x2(v1.x*v2.x,v1.x*v2.y,v1.y*v2.x,v1.y*v2.y);
    };

inline void printMatrix(Matrix2x2 &m)
    {
    cout << endl << m.x11 << "    " << m.x12 << endl <<m.x21<<"    "<<m.x22<<endl<<endl;
    };

#undef HOSTDEVICE
#endif
