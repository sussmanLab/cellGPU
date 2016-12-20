#ifndef STDINCLUDE
#define STDINCLUDE

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#define THRESHOLD 1e-18
#define EPSILON 1e-18

#include <cmath>
#include <algorithm>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/time.h>

using namespace std;


#include "vector_types.h"
#include "vector_functions.h"

#define PI 3.14159265358979323846


//decide whether to compute everything in floating point or double precision
#ifndef SCALARFLOAT
//doubles
#define Dscalar double
#define Dscalar2 double2
#define Dscalar4 double4
#define ncDscalar ncDouble
#define cur_norm curand_normal_double
#define Cos cos
#define Sin sin
#define Floor floor
#define Ceil ceil

#else
//floats

#define Dscalar float
#define Dscalar2 float2
#define Dscalar4 float4
#define ncDscalar ncFloat
#define cur_norm curand_normal
#define Cos cosf
#define Sin sinf
#define Floor floorf
#define Ceil ceilf
#endif

HOSTDEVICE bool operator<(const Dscalar2 &a, const Dscalar2 &b)
    {
    return a.x<b.x;
    }

HOSTDEVICE bool operator==(const Dscalar2 &a, const Dscalar2 &b)
    {
    return (a.x==b.x &&a.y==b.y);
    }

HOSTDEVICE Dscalar2 make_Dscalar2(Dscalar x, Dscalar y)
    {
    Dscalar2 ans;
    ans.x =x;
    ans.y=y;
    return ans;
    }

HOSTDEVICE Dscalar2 operator+(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x+b.x,a.y+b.y);
    }

HOSTDEVICE Dscalar2 operator-(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x-b.x,a.y-b.y);
    }

HOSTDEVICE Dscalar4 make_Dscalar4(Dscalar x, Dscalar y,Dscalar z, Dscalar w)
    {
    Dscalar4 ans;
    ans.x =x;
    ans.y=y;
    ans.z =z;
    ans.w=w;
    return ans;
    }



#undef HOSTDEVICE

#endif

