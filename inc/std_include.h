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
#include "deprecated.h"

#define PI 3.14159265358979323846

//decide whether to compute everything in floating point or double precision
#ifndef SCALARFLOAT
//double variables types
#define Dscalar double
#define Dscalar2 double2
#define Dscalar3 double3
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
#define Dscalar3 float3
#define Dscalar4 float4
#define ncDscalar ncFloat
#define cur_norm curand_normal
#define Cos cosf
#define Sin sinf
#define Floor floorf
#define Ceil ceilf
#endif

//!Less than operator for Dscalars just sorts by the x-coordinate
HOSTDEVICE bool operator<(const Dscalar2 &a, const Dscalar2 &b)
    {
    return a.x<b.x;
    }

//!Equality operator tests for.... equality of both elements
HOSTDEVICE bool operator==(const Dscalar2 &a, const Dscalar2 &b)
    {
    return (a.x==b.x &&a.y==b.y);
    }

//!return a Dscalar2 from two Dscalars
HOSTDEVICE Dscalar2 make_Dscalar2(Dscalar x, Dscalar y)
    {
    Dscalar2 ans;
    ans.x =x;
    ans.y=y;
    return ans;
    }

//!component-wise addition of two Dscalar2s
HOSTDEVICE Dscalar2 operator+(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x+b.x,a.y+b.y);
    }

//!component-wise subtraction of two Dscalar2s
HOSTDEVICE Dscalar2 operator-(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x-b.x,a.y-b.y);
    }

//!multiplication of Dscalar2 by Dscalar
HOSTDEVICE Dscalar2 operator*(const Dscalar &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a*b.x,a*b.y);
    }

//!print a Dscalar2 to screen
inline __attribute__((always_inline)) void printDscalar2(Dscalar2 a)
    {
    printf("%f\t%f\n",a.x,a.y);
    };


//!return a Dscalar3 from three Dscalars
HOSTDEVICE Dscalar3 make_Dscalar3(Dscalar x, Dscalar y,Dscalar z)
    {
    Dscalar3 ans;
    ans.x =x;
    ans.y=y;
    ans.z =z;
    return ans;
    }

//!return a Dscalar4 from four Dscalars
HOSTDEVICE Dscalar4 make_Dscalar4(Dscalar x, Dscalar y,Dscalar z, Dscalar w)
    {
    Dscalar4 ans;
    ans.x =x;
    ans.y=y;
    ans.z =z;
    ans.w=w;
    return ans;
    }

//!Handle errors in kernel calls...returns file and line numbers if cudaSuccess doesn't pan out
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

//!A utility function for checking if a file exists
inline bool fileExists(const std::string& name)
    {
    ifstream f(name.c_str());
    return f.good();
    }

//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        cout << "Requested GPU (device " << USE_GPU<<") does not exist. Stopping triangulation" << endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)    cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        if (verbose)
            {
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        };
    if (!verbose)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    };

//!calls the above function, setting the cuda-capable device corresponding to the integer passed in (or returning false if USE_GPU < 0)
__host__ inline bool setCudaDevice(int USE_GPU,bool verbose = false)
    {
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU,verbose);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        return true;
        }
    else
        return false;
    };

//A macro to wrap cuda calls
#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))
//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);

#undef HOSTDEVICE
#endif
