#ifndef STDINCLUDE
#define STDINCLUDE

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#include "vector_types.h"
#include "vector_functions.h"

#define PI 3.14159265358979323846

#define Dscalar double
#define Dscalar2 double2
#define Dscalar4 double4
#define ncDscalar ncDouble

//#define cur_norm curand_normal_double
//#define cur_norm curand_normal

HOSTDEVICE bool operator<(const Dscalar2 &a, const Dscalar2 &b)
    {
    return a.x<b.x;
    }

HOSTDEVICE Dscalar2 make_Dscalar2(Dscalar x, Dscalar y)
    {
    Dscalar2 ans;
    ans.x =x;
    ans.y=y;
    return ans;
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

