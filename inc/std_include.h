#ifndef STDINCLUDE
#define STDINCLUDE

/*! \file std_include.h
a file of bad practice to be included all the time... carries with it things DMS often uses
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

#include "vector_types.h"
#include "vector_functions.h"

#define PI 3.14159265358979323846

//decide whether to compute everything in floating point or double precision
//double variables types
//the cuda RNG
#define cur_norm curand_normal_double
//trig and special funtions
#define Cos cos
#define Sin sin
#define Floor floor
#define Ceil ceil


//!Less than operator for doubles just sorts by the x-coordinate
HOSTDEVICE bool operator<(const double2 &a, const double2 &b)
    {
    return a.x<b.x;
    }

//!Equality operator tests for.... equality of both elements
HOSTDEVICE bool operator==(const double2 &a, const double2 &b)
    {
    return (a.x==b.x &&a.y==b.y);
    }

//!component-wise addition of two double2s
HOSTDEVICE double2 operator+(const double2 &a, const double2 &b)
    {
    return make_double2(a.x+b.x,a.y+b.y);
    }

//!component-wise subtraction of two double2s
HOSTDEVICE double2 operator-(const double2 &a, const double2 &b)
    {
    return make_double2(a.x-b.x,a.y-b.y);
    }

//!multiplication of double2 by double
HOSTDEVICE double2 operator*(const double &a, const double2 &b)
    {
    return make_double2(a*b.x,a*b.y);
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

//!Report somewhere that code needs to be written
static void unwrittenCode(const char *message, const char *file, int line)
    {
    printf("\nCode unwritten (file %s; line %d)\nMessage: %s\n",file,line,message);
    throw std::exception();
    }

//A macro to wrap cuda calls
#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))
//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);
//A macro to say code needs to be written
#define UNWRITTENCODE(message) (unwrittenCode(message,__FILE__,__LINE__))

#undef HOSTDEVICE
#endif
