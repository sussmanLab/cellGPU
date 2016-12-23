#ifndef HILBERTSORT
#define HILBERTSORT


#include "std_include.h"
#include <cmath>
#include <vector>
#include "cuda.h"
#include "vector_types.h"
#include "vector_functions.h"


#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


/*!
This structure can help sort scalar2's according to their position along a hilbert curve of order M...
This sorting can improve data locality (i.e. particles that are close to each other in physical space reside
close to each other in memory). This is a small boost for CPU-based code, but can be very important
for the efficiency of GPU-based execution.
Some of the code here is straight from wikipedia!
*/
struct HilbertSorter
    {
    private:
        gpubox box; //!<A box to put the particles in the unit square for easy sorting
        int M;      //!<The integer order of the Hilbert curve to use
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

            //!Rotate...from wikipedia
            HOSTDEVICE void HilbertRotate(int n, int &x, int &y, int rx, int ry)
                {
                int t;
                if (ry == 0)
                    {
                    //reflect
                    if(ry == 1)
                        {
                        x = n-1-x;
                        y=n-1-y;
                        };
                    //flip
                    t = x;
                    x = y;
                    y = t;
                    };
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

            //The following bit-operating code is from wikipedia
            int d = 0;
            int  rx,ry;
            for (int s = n/2; s > 0; s= s/2)
                {
                rx = ( x & s) > 0;
                ry = ( y & s) > 0;
                d = d + s * s * ( ( 3* rx ) ^ ry);
                HilbertRotate(s,x,y,rx,ry);
                };

            return d;
            };
    };

#undef HOSTDEVICE

#endif
