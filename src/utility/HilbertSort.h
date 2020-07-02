#ifndef HILBERTSORT
#define HILBERTSORT

/*
Some code in the HilbertRotate and getIdx functions calls John Burkardt's HILBERT_CURVE code:
https://people.sc.fsu.edu/~jburkardt/cpp_src/hilbert_curve/hilbert_curve.html
which is released under the GNU LGPL license
*/

#include "std_include.h"
#include "hilbert_curve.hpp"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file HilbertSort.h */
//!Spatially sort points in 2D according to a 1D Hilbert curve
/*!
This structure can help sort scalar2's according to their position along a hilbert curve of order M...
This sorting can improve data locality (i.e. particles that are close to each other in physical space reside
close to each other in memory). This is a small boost for CPU-based code, but can be very important
for the efficiency of GPU-based execution.
*/
struct HilbertSorter
    {
    public:
        //!The only constructor requires a box
        HOSTDEVICE HilbertSorter(gpubox Box)
            {
            Dscalar x11,x12,x21,x22;
            Box.getBoxDims(x11,x12,x21,x22);
            box.setGeneral(x11,x12,x21,x22);

            int mm = 1;
            int temp = 2;
            while ((Dscalar)temp < x11)
                {
                temp *=2;
                mm +=1;
                };
            setOrder((int)min(30,mm+4));
            }

        gpubox box; //!<A box to put the particles in the unit square for easy sorting
        int M;      //!<The integer order of the Hilbert curve to use
        //some functions to help out...

        //!Set the order of the desired HC
        HOSTDEVICE void setOrder(int m){M=m;};

        //!A hand-written function to take integer powers of integers
        HOSTDEVICE int int_power(int i, int j)
            {
            int value;
            if (j < 0)
                {
                if (i == 1)
                    value = 1;
                else
                    value = 0;
                }
            else if (j == 0)
                value = 1;
            else if (j ==1)
                value = i;
            else
                {
                value = 1;
                for (int k = 1; k <=j; ++k)
                    value = value*i;
                };

            return value;
            };

            //!Rotate/flip quadrants appropriately
            HOSTDEVICE void HilbertRotate(int n, int &x, int &y, int rx, int ry)
                {
                //call Burkardt code...this is no longer needed, and can be deprecated
                rot(n,x,y,rx,ry);
                return;
                };

        //!Convert a real(x,y) pair to a nearby integer pair, and then gets the 1D Hilbert coordinate of that point.
        //!The number of cells is 2^M (M is the index of the HC))
        HOSTDEVICE int getIdx(Dscalar2 point)
            {

            //x and y need to be in the range 0 <= x,y < n, where n=2^M
            Dscalar2 virtualPos;
            box.invTrans(point,virtualPos);

            int n = int_power(2,M);
            int x,y;
            x = (int) floor(n*virtualPos.x);
            y = (int) floor(n*virtualPos.y);

            //call Burkardt code
            int d = xy2d(M,x,y);
            return d;
            };
    };

#undef HOSTDEVICE
#endif
