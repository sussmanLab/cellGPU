#ifndef Matrix_H
#define Matrix_H

#include "vector_types.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


struct Matrix2x2
    {//contains a {{x11,x12},{x21,x22}} set, and matrix manipulations
    private:

    public:
        float x11, x12, x21, x22;
        HOSTDEVICE Matrix2x2() : x11(1.0), x12(0.0), x21(0.0),x22(1.0) {};
        HOSTDEVICE Matrix2x2(float y11, float y12, float y21,float y22) : x11(y11), x12(y12), x21(y21),x22(y22) {};


        HOSTDEVICE void set(float y11, float y12, float y21, float y22)
                            {
                            x11=y11; x12=y12;x21=y21;x22=y22;
                            };

        //assignment
        HOSTDEVICE void operator=(const Matrix2x2 &m2)
                            {
                            set(m2.x11,m2.x12,m2.x21,m2.x22);
                            };
        //matrix multiplication
        HOSTDEVICE void operator*=(const Matrix2x2 &m2)
                            {
                            set(x11*m2.x11 + x12*m2.x21,
                                x11*m2.x12 + x12*m2.x22,
                                x21*m2.x11 + x22*m2.x21,
                                x21*m2.x12 + x22*m2.x22
                                );
                            };
        HOSTDEVICE friend Matrix2x2 operator*(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp*=m2;
                            return temp;
                            };

        //scalar multiplication
        HOSTDEVICE void operator*=(float a)
                            {
                            set(a*x11,a*x12,a*x21,a*x22);
                            };
        HOSTDEVICE friend Matrix2x2 operator*(const float a, const Matrix2x2 &m)
                            {
                            Matrix2x2 temp(m);
                            temp*=a;
                            return temp;
                            };

        //Matrix addition
        HOSTDEVICE void operator+=(const Matrix2x2 &m2)
                            {
                            set(x11+m2.x11,
                                x12+m2.x12,
                                x21+m2.x21,
                                x22+m2.x22
                               );
                            };
        HOSTDEVICE friend Matrix2x2 operator+(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp+=m2;
                            return temp;
                            };


        HOSTDEVICE friend float2 operator*(const Matrix2x2 &m, const float2 &v)
                            {
                            float2 temp;
                            temp.x = m.x11*v.x+m.x12*v.y;
                            temp.y = m.x21*v.x+m.x22*v.y;
                            return temp;
                            };

    };

HOSTDEVICE Matrix2x2 dyad(float2 &v1, float2 &v2)
    {
    return Matrix2x2(v1.x*v2.x,v1.x*v2.y,v1.y*v2.x,v1.y*v2.y);
    };

#endif
