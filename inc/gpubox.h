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

struct gpubox
    {
    private:
        Dscalar x11,x12,x21,x22;//the transformation matrix defining the box
        Dscalar xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;


    public:
        HOSTDEVICE gpubox(){isSquare = false;};
        HOSTDEVICE gpubox(Dscalar x, Dscalar y){setSquare(x,y);};
        HOSTDEVICE gpubox(Dscalar a, Dscalar b, Dscalar c, Dscalar d){setGeneral(a,b,c,d);};
        HOSTDEVICE void getBoxDims(Dscalar &xx, Dscalar &xy, Dscalar &yx, Dscalar &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};
        HOSTDEVICE void getBoxInvDims(Dscalar &xx, Dscalar &xy, Dscalar &yx, Dscalar &yy)
            {xx=xi11;xy=xi12;yx=xi21;yy=xi22;};

        HOSTDEVICE void setSquare(Dscalar x, Dscalar y);
        HOSTDEVICE void setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d);

        HOSTDEVICE void putInBoxReal(Dscalar2 &p1);
        HOSTDEVICE void putInBox(Dscalar2 &vp);
        HOSTDEVICE void Trans(const Dscalar2 &p1, Dscalar2 &pans);
        HOSTDEVICE void invTrans(const Dscalar2 p1, Dscalar2 &pans);
        HOSTDEVICE void minDist(const Dscalar2 &p1, const Dscalar2 &p2, Dscalar2 &pans);

        HOSTDEVICE void move(Dscalar2 &p1, const Dscalar2 &disp);

        HOSTDEVICE void operator=(gpubox &other)
            {
            Dscalar b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    };

void gpubox::setSquare(Dscalar x, Dscalar y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    };

void gpubox::setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d)
    {
    x11=a;x12=b;x21=c;x22=d;
    xi11 = 1./x11;xi22=1./x22;
    Dscalar prefactor = 1.0/(a*d-b*c);
    if(fabs(prefactor)>0)
        {
        xi11=prefactor*d;
        xi12=-prefactor*b;
        xi21=-prefactor*c;
        xi22=prefactor*a;
        };
    isSquare = false;
    };

void gpubox::Trans(const Dscalar2 &p1, Dscalar2 &pans)
    {
    pans.x = x11*p1.x + x12*p1.y;
    pans.y = x21*p1.x + x22*p1.y;
    };

void gpubox::invTrans(const Dscalar2 p1, Dscalar2 &pans)
    {
    pans.x = xi11*p1.x + xi12*p1.y;
    pans.y = xi21*p1.x + xi22*p1.y;
    };

void gpubox::putInBoxReal(Dscalar2 &p1)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    Dscalar2 vP;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

void gpubox::putInBox(Dscalar2 &vp)
    {//acts on points in the virtual space
    while(vp.x < 0) vp.x +=1.0;
    while(vp.y < 0) vp.y +=1.0;

    while(vp.x>=1.0)
        {
        vp.x -= 1.0;
        };
    while(vp.y>=1.)
        {
        vp.y -= 1.;
        };
    };

void gpubox::minDist(const Dscalar2 &p1, const Dscalar2 &p2, Dscalar2 &pans)
    {
    Dscalar2 vA,vB;
    invTrans(p1,vA);
    invTrans(p2,vB);
    Dscalar2 disp= make_Dscalar2(vA.x-vB.x,vA.y-vB.y);

    while(fabs(disp.x)>0.5)
        {
        Dscalar sgn = (disp.x > 0) - (disp.x < 0);
        disp.x = disp.x - sgn;
        };
    while(fabs(disp.y)>0.5)
        {
        Dscalar sgn = (disp.y > 0) - (disp.y < 0);
        disp.y = disp.y - sgn;
        };

    Trans(disp,pans);
    };

void gpubox::move(Dscalar2 &p1, const Dscalar2 &disp)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    Dscalar2 vP;
    p1.x = p1.x+disp.x;
    p1.y = p1.y+disp.y;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE

#endif

