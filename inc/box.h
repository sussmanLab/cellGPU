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
#include "structures.h"


namespace voroguppy
{
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


class box
    {
    private:
        dbl x11,x12,x21,x22;//the transformation matrix defining the box
        dbl xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;

    public:
        HOSTDEVICE box(){isSquare = false;};
        HOSTDEVICE box(dbl x, dbl y){setSquare(x,y);};
        HOSTDEVICE box(dbl a, dbl b, dbl c, dbl d){setGeneral(a,b,c,d);};
        HOSTDEVICE void getBoxDims(dbl &xx, dbl &xy, dbl &yx, dbl &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};

        HOSTDEVICE void setSquare(dbl x, dbl y);
        HOSTDEVICE void setGeneral(dbl a, dbl b,dbl c, dbl d);

        HOSTDEVICE void putInBox(pt &vp);
        HOSTDEVICE void Trans(const pt &p1, pt &pans);
        HOSTDEVICE void invTrans(const pt &p1, pt &pans);
        HOSTDEVICE void minDist(const pt &p1, const pt &p2, pt &pans);

        HOSTDEVICE void move(pt &p1, const pt &disp);

        HOSTDEVICE void operator=(box &other)
            {
            dbl b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    };

void box::setSquare(dbl x, dbl y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    };

void box::setGeneral(dbl a, dbl b,dbl c, dbl d)
    {
    x11=a;x12=b;x21=c;x22=d;
    xi11 = 1./x11;xi22=1./x22;
    dbl prefactor = 1.0/(a*d-b*c);
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
        dbl sgn = (vp.x > 0) - (vp.x < 0);
        vp.x = vp.x - sgn;
        };
    while(fabs(vp.y)>=1.)
        {
        dbl sgn = (vp.y > 0) - (vp.y < 0);
        vp.y = vp.y - sgn;
        };
    };

void box::minDist(const pt &p1, const pt &p2, pt &pans)
    {
    pt vA,vB;
    invTrans(p1,vA);
    invTrans(p2,vB);
    //this function is called a lot...so, optimize.
    //structures::pt disp;
    //disp.x = xi11*p1.x+xi12*p1.y - xi11*p2.x+xi12*p2.y;
    //disp.y = xi21*p1.x+xi22*p1.y - xi21*p2.x+xi22*p2.y;
    pt disp=vA-vB;
    while(fabs(disp.x)>0.5)
        {
        dbl sgn = (disp.x > 0) - (disp.x < 0);
        disp.x = disp.x - sgn;
        };
    while(fabs(disp.y)>0.5)
        {
        dbl sgn = (disp.y > 0) - (disp.y < 0);
        disp.y = disp.y - sgn;
        };
    //pans.x = x11*disp.x + x12*disp.y;
    //pans.y = x21*disp.x + x22*disp.y;
    Trans(disp,pans);
    };

/*
void box::minDist(const pt &p1, const pt &p2, pt &pans)
    {
    float dx,dy;
    dx = xi11*p1.x+xi12*p1.y - xi11*p2.x+xi12*p2.y;
    dy = xi21*p1.x+xi22*p1.y - xi21*p2.x+xi22*p2.y;
    float sgn;
    while(fabs(dx)>0.5)
        {
        sgn = (dx > 0) - (dx < 0);
        dx -= sgn;
        };
    while(fabs(dy)>0.5)
        {
        sgn = (dy > 0) - (dy < 0);
        dy -= sgn;
        };
    pans.x = x11*dx + x12*dy;
    pans.y = x21*dx + x22*dy;
    };
*/

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
}

#endif
