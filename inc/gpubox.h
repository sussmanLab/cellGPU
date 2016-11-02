#ifndef GPUBOX
#define GPUBOX

///////////////////////
//Periodic Box for the gpu
//
//Description
//      This is a class that defines periodic boundary conditions in 2d
//
//Implements
//      Computes minimal distances using periodic boundary conditions.
//      Displaces particles while respecting the periodic boundary conditions.
using namespace std;
#include <cmath>
#include <vector>
#include "cuda.h"
#include "vector_types.h"
#include "vector_functions.h"


namespace voroguppy
{
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#define dbl float
struct gpubox
    {
    private:
        float x11,x12,x21,x22;//the transformation matrix defining the box
        float xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;


    public:
        HOSTDEVICE gpubox(){isSquare = false;};
        HOSTDEVICE gpubox(float x, float y){setSquare(x,y);};
        HOSTDEVICE gpubox(float a, float b, float c, float d){setGeneral(a,b,c,d);};
        HOSTDEVICE void getBoxDims(float &xx, float &xy, float &yx, float &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};
        HOSTDEVICE void getBoxInvDims(float &xx, float &xy, float &yx, float &yy)
            {xx=xi11;xy=xi12;yx=xi21;yy=xi22;};

        HOSTDEVICE void setSquare(float x, float y);
        HOSTDEVICE void setGeneral(float a, float b,float c, float d);

        HOSTDEVICE void putInBoxReal(float2 &p1);
        HOSTDEVICE void putInBox(float2 &vp);
        HOSTDEVICE void Trans(const float2 &p1, float2 &pans);
        HOSTDEVICE void invTrans(const float2 &p1, float2 &pans);
        HOSTDEVICE void minDist(const float2 &p1, const float2 &p2, float2 &pans);

        HOSTDEVICE void move(float2 &p1, const float2 &disp);

        HOSTDEVICE void operator=(gpubox &other)
            {
            float b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    };

void gpubox::setSquare(float x, float y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    };

void gpubox::setGeneral(float a, float b,float c, float d)
    {
    x11=a;x12=b;x21=c;x22=d;
    xi11 = 1./x11;xi22=1./x22;
    float prefactor = 1.0/(a*d-b*c);
    if(fabs(prefactor)>0)
        {
        xi11=prefactor*d;
        xi12=-prefactor*b;
        xi21=-prefactor*c;
        xi22=prefactor*a;
        };
    isSquare = false;
    };

void gpubox::Trans(const float2 &p1, float2 &pans)
    {
    pans.x = x11*p1.x + x12*p1.y;
    pans.y = x21*p1.x + x22*p1.y;
    };

void gpubox::invTrans(const float2 &p1, float2 &pans)
    {
    pans.x = xi11*p1.x + xi12*p1.y;
    pans.y = xi21*p1.x + xi22*p1.y;
    };

void gpubox::putInBoxReal(float2 &p1)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    float2 vP;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

void gpubox::putInBox(float2 &vp)
    {//acts on points in the virtual space
    if(vp.x < 0) vp.x +=1.0;
    if(vp.y < 0) vp.y +=1.0;

    while(fabs(vp.x)>=1.0)
        {
        float sgn = (vp.x > 0) - (vp.x < 0);
        vp.x = vp.x - sgn;
        };
    while(fabs(vp.y)>=1.)
        {
        float sgn = (vp.y > 0) - (vp.y < 0);
        vp.y = vp.y - sgn;
        };
    };

void gpubox::minDist(const float2 &p1, const float2 &p2, float2 &pans)
    {
    float2 vA,vB;
    invTrans(p1,vA);
    invTrans(p2,vB);
    //this function is called a lot...so, optimize.
    //disp.x = xi11*p1.x+xi12*p1.y - xi11*p2.x+xi12*p2.y;
    //disp.y = xi21*p1.x+xi22*p1.y - xi21*p2.x+xi22*p2.y;
    float2 disp= make_float2(vA.x-vB.x,vA.y-vB.y);
    while(fabs(disp.x)>0.5)
        {
        float sgn = (disp.x > 0) - (disp.x < 0);
        disp.x = disp.x - sgn;
        };
    while(fabs(disp.y)>0.5)
        {
        float sgn = (disp.y > 0) - (disp.y < 0);
        disp.y = disp.y - sgn;
        };
    //pans.x = x11*disp.x + x12*disp.y;
    //pans.y = x21*disp.x + x22*disp.y;
    Trans(disp,pans);
    };

void gpubox::move(float2 &p1, const float2 &disp)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    float2 vP;
    p1.x = p1.x+disp.x;
    p1.y = p1.y+disp.y;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE
}

#endif

