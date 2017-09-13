#ifndef GPUBOX
#define GPUBOX

#include "std_include.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file gpubox.h */
//!A simple box defining a 2D periodic domain
/*!
gpubox  periodic boundary conditions in 2D, computing minimum distances between
periodic images, displacing particles and putting them back in the central unit cell, etc.
The workhorse of this class is calling
Box.minDist(vecA,vecB,&disp),
which computes the displacement between vecA and the closest periodic image of vecB and
stores the result in disp. Also
Box.putInBoxReal(&point), which will take the point and put it back in the primary unit cell.
Please note that while the gpubox class can handle generic 2D periodic domains, many of the other classes
that interface with it do not yet have this functionality implemented.
*/
struct gpubox
    {
    public:
        HOSTDEVICE gpubox(){isSquare = false;};
        //!Construct a rectangular box containing the unit cell ((0,0),(x,0),(x,y),(0,y))
        HOSTDEVICE gpubox(Dscalar x, Dscalar y){setSquare(x,y);};
        //!Construct a non-rectangular box
        HOSTDEVICE gpubox(Dscalar a, Dscalar b, Dscalar c, Dscalar d){setGeneral(a,b,c,d);};
        //!Get the dimensions of the box
        HOSTDEVICE void getBoxDims(Dscalar &xx, Dscalar &xy, Dscalar &yx, Dscalar &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};
        //!Check if the box is rectangular or not (as certain optimizations can then be used)
        HOSTDEVICE bool isBoxSquare(){return isSquare;};
        //!Get the inverse of the box transformation matrix
        HOSTDEVICE void getBoxInvDims(Dscalar &xx, Dscalar &xy, Dscalar &yx, Dscalar &yy)
            {xx=xi11;xy=xi12;yx=xi21;yy=xi22;};

        //!Set the box to some new rectangular specification
        HOSTDEVICE void setSquare(Dscalar x, Dscalar y);
        //!Set the box to some new generic specification
        HOSTDEVICE void setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d);

        //!Take the point and put it back in the unit cell
        HOSTDEVICE void putInBoxReal(Dscalar2 &p1);
        //! Take a point in the unit square and find its position in the box
        HOSTDEVICE void Trans(const Dscalar2 &p1, Dscalar2 &pans);
        //! Take a point in the box and find its position in the unit square
        HOSTDEVICE void invTrans(const Dscalar2 p1, Dscalar2 &pans);
        //!Calculate the minimum distance between two points
        HOSTDEVICE void minDist(const Dscalar2 &p1, const Dscalar2 &p2, Dscalar2 &pans);
        //!Move p1 by the amount disp, then put it in the box
        HOSTDEVICE void move(Dscalar2 &p1, const Dscalar2 &disp);

        HOSTDEVICE void operator=(gpubox &other)
            {
            Dscalar b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    protected:
        //!The transformation matrix defining the periodic box
        Dscalar x11,x12,x21,x22;//the transformation matrix defining the box
        Dscalar halfx11,halfx22;
        //!The inverse of the transformation matrix
        Dscalar xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;

        HOSTDEVICE void putInBox(Dscalar2 &vp);
    };

void gpubox::setSquare(Dscalar x, Dscalar y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    halfx11 = x11*0.5;
    halfx22 = x22*0.5;
    };

void gpubox::setGeneral(Dscalar a, Dscalar b,Dscalar c, Dscalar d)
    {
    x11=a;x12=b;x21=c;x22=d;
    xi11 = 1./x11;xi22=1./x22;
    Dscalar prefactor = 1.0/(a*d-b*c);
    halfx11 = x11*0.5;
    halfx22 = x22*0.5;
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
    if(isSquare)
        {
        pans.x = x11*p1.x;
        pans.y = x22*p1.y;
        }
    else
        {
        pans.x = x11*p1.x + x12*p1.y;
        pans.y = x21*p1.x + x22*p1.y;
        };
    };

void gpubox::invTrans(const Dscalar2 p1, Dscalar2 &pans)
    {
    if(isSquare)
        {
        pans.x = xi11*p1.x;
        pans.y = xi22*p1.y;
        }
    else
        {
        pans.x = xi11*p1.x + xi12*p1.y;
        pans.y = xi21*p1.x + xi22*p1.y;
        };
    };

void gpubox::putInBoxReal(Dscalar2 &p1)
    {//assume real space entries. Puts it back in box
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
    if (isSquare)
        {
        pans.x = p1.x-p2.x;
        pans.y = p1.y-p2.y;;
        while(pans.x < -halfx11) pans.x += x11;
        while(pans.y < -halfx22) pans.y += x22;
        while(pans.x > halfx11) pans.x -= x11;
        while(pans.y > halfx22) pans.y -= x22;
        }
    else
        {
        Dscalar2 vA,vB;
        invTrans(p1,vA);
        invTrans(p2,vB);
        Dscalar2 disp= make_Dscalar2(vA.x-vB.x,vA.y-vB.y);

        while(disp.x < -0.5) disp.x +=1.0;
        while(disp.y < -0.5) disp.y +=1.0;
        while(disp.x > 0.5) disp.x -=1.0;
        while(disp.y > 0.5) disp.y -=1.0;

        Trans(disp,pans);
        };
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

typedef shared_ptr<gpubox> BoxPtr;

#undef HOSTDEVICE
#endif
