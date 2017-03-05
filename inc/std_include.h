#ifndef STDINCLUDE
#define STDINCLUDE

/*! \file std_include.h
a file to be included all the time... carries with it things DMS often uses
Crucially, it also defines Dscalars as either floats or doubles, depending on
how the program is compiled
*/

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#define THRESHOLD 1e-18
#define EPSILON 1e-18

#include <cmath>
#include <algorithm>
#include <memory>
#include <ctype.h>
#include <random>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/time.h>
#include <string.h>
#include <stdexcept>
#include <cassert>

using namespace std;


#include <cuda_runtime.h>
#include "vector_types.h"
#include "vector_functions.h"

#define PI 3.14159265358979323846

//decide whether to compute everything in floating point or double precision
#ifndef SCALARFLOAT
//double variables types
#define Dscalar double
#define Dscalar2 double2
#define Dscalar4 double4
//the netcdf variable type
#define ncDscalar ncDouble
//the cuda RNG
#define cur_norm curand_normal_double
//trig and special funtions
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

//handle errors in kernel calls
static void HandleError(cudaError_t err, const char *file, int line)
    {
    //as an additional debugging check, if always synchronize cuda threads after every kernel call
    #ifdef CUDATHREADSYNC
    cudaThreadSynchronize();
    #endif
    if (err != cudaSuccess)
        {
        printf("\nError: %s in file %s at line %d\n",cudaGetErrorString(err),file,line);
        throw std::exception();
        }
    }

#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))

#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);
#undef HOSTDEVICE

#endif
