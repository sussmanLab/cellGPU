#ifndef PERIODICBOX
#define PERIODICBOX

///////////////////////
//Periodic Box class.
//
//Description
//      This is a class that defines periodic boundary conditions in 2d
//
//Implements
//      Computes minimal distances using periodic boundary conditions.
//      Displaces particles while respecting the periodic boundary conditions.
using namespace std;
#include "std_include.h"
#include "structures.h"


#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


class box
    {
    private:
        Dscalar x11,x12,x21,x22;//the transformation matrix defining the box
        Dscalar xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;

    public:
        HOSTDEVICE box(){isSquare = false;};
        HOSTDEVICE box(Dscalar x, Dscalar y){setSquare(x,y);};
        HOSTDEVICE box(Dscalar a, Dscalar b, Dscalar c, Dscalar d){setGeneral(a,b,c,d);};
        HOSTDEVICE void getBoxDims(Dscalar &xx, Dscalar &xy, Dscalar &yx, Dscalar &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};

        HOSTDEVICE void setSquare(Dscalar x, Dscalar y);
        HOSTDEVICE void setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d);

        HOSTDEVICE void putInBox(pt &vp);
        HOSTDEVICE void Trans(const pt &p1, pt &pans);
        HOSTDEVICE void invTrans(const pt &p1, pt &pans);
        HOSTDEVICE void minDist(const pt &p1, const pt &p2, pt &pans);

        HOSTDEVICE void move(pt &p1, const pt &disp);

        HOSTDEVICE void operator=(box &other)
            {
            Dscalar b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    };

void box::setSquare(Dscalar x, Dscalar y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    };

void box::setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d)
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

void box::Trans(const pt &p1, pt &pans)
    {
    pans.x = x11*p1.x + x12*p1.y;
    pans.y = x21*p1.x + x22*p1.y;
    };

void box::invTrans(const pt &p1, pt &pans)
    {
    pans.x = xi11*p1.x + xi12*p1.y;
    pans.y = xi21*p1.x + xi22*p1.y;
    };

void box::putInBox(pt &vp)
    {//acts on points in the virtual space
    while(fabs(vp.x)>=1.0)
        {
        Dscalar sgn = (vp.x > 0) - (vp.x < 0);
        vp.x = vp.x - sgn;
        };
    while(fabs(vp.y)>=1.)
        {
        Dscalar sgn = (vp.y > 0) - (vp.y < 0);
        vp.y = vp.y - sgn;
        };
    };

void box::minDist(const pt &p1, const pt &p2, pt &pans)
    {
    pt vA,vB;
    invTrans(p1,vA);
    invTrans(p2,vB);
    pt disp=vA-vB;
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


void box::move(pt &p1, const pt &disp)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    pt vP,vD;
    p1 = p1+disp;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE

#endif
