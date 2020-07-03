#ifndef PERIODICBOX_H
#define PERIODICBOX_H

#include "std_include.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file periodicBoundaries.h */
//!A simple box defining a 2D periodic domain
/*!
periodicBoundaries  periodic boundary conditions in 2D, computing minimum distances between
periodic images, displacing particles and putting them back in the central unit cell, etc.
The workhorse of this class is calling
Box.minDist(vecA,vecB,&disp),
which computes the displacement between vecA and the closest periodic image of vecB and
stores the result in disp. Also
Box.putInBoxReal(&point), which will take the point and put it back in the primary unit cell.
Please note that while the periodicBoundaries class can handle generic 2D periodic domains, many of the other classes
that interface with it do not yet have this functionality implemented.
*/
struct periodicBoundaries
    {
    public:
        HOSTDEVICE periodicBoundaries(){isSquare = false;};
        //!Construct a rectangular box containing the unit cell ((0,0),(x,0),(x,y),(0,y))
        HOSTDEVICE periodicBoundaries(double x, double y){setSquare(x,y);};
        //!Construct a non-rectangular box
        HOSTDEVICE periodicBoundaries(double a, double b, double c, double d){setGeneral(a,b,c,d);};
        //!Get the dimensions of the box
        HOSTDEVICE void getBoxDims(double &xx, double &xy, double &yx, double &yy)
            {xx=x11;xy=x12;yx=x21;yy=x22;};
        //!Check if the box is rectangular or not (as certain optimizations can then be used)
        HOSTDEVICE bool isBoxSquare(){return isSquare;};
        //!Get the inverse of the box transformation matrix
        HOSTDEVICE void getBoxInvDims(double &xx, double &xy, double &yx, double &yy)
            {xx=xi11;xy=xi12;yx=xi21;yy=xi22;};

        //!Set the box to some new rectangular specification
        HOSTDEVICE void setSquare(double x, double y);
        //!Set the box to some new generic specification
        HOSTDEVICE void setGeneral(double a, double b,double c, double d);

        //!Take the point and put it back in the unit cell
        HOSTDEVICE void putInBoxReal(double2 &p1);
        //! Take a point in the unit square and find its position in the box
        HOSTDEVICE void Trans(const double2 &p1, double2 &pans);
        //! Take a point in the box and find its position in the unit square
        HOSTDEVICE void invTrans(const double2 p1, double2 &pans);
        //!Calculate the minimum distance between two points
        HOSTDEVICE void minDist(const double2 &p1, const double2 &p2, double2 &pans);
        //!Move p1 by the amount disp, then put it in the box
        HOSTDEVICE void move(double2 &p1, const double2 &disp);

        HOSTDEVICE void operator=(periodicBoundaries &other)
            {
            double b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
    protected:
        //!The transformation matrix defining the periodic box
        double x11,x12,x21,x22;//the transformation matrix defining the box
        double halfx11,halfx22;
        //!The inverse of the transformation matrix
        double xi11,xi12,xi21,xi22;//it's inverse
        bool isSquare;

        HOSTDEVICE void putInBox(double2 &vp);
    };

void periodicBoundaries::setSquare(double x, double y)
    {
    x11=x;x22=y;
    x12=0.0;x21=0.0;
    xi11 = 1./x11;xi22=1./x22;
    xi12=0.0;xi21=0.0;
    isSquare = true;
    halfx11 = x11*0.5;
    halfx22 = x22*0.5;
    };

void periodicBoundaries::setGeneral(double a, double b,double c, double d)
    {
    x11=a;x12=b;x21=c;x22=d;
    xi11 = 1./x11;xi22=1./x22;
    double prefactor = 1.0/(a*d-b*c);
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

void periodicBoundaries::Trans(const double2 &p1, double2 &pans)
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

void periodicBoundaries::invTrans(const double2 p1, double2 &pans)
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

void periodicBoundaries::putInBoxReal(double2 &p1)
    {//assume real space entries. Puts it back in box
    double2 vP;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

void periodicBoundaries::putInBox(double2 &vp)
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

void periodicBoundaries::minDist(const double2 &p1, const double2 &p2, double2 &pans)
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
        double2 vA,vB;
        invTrans(p1,vA);
        invTrans(p2,vB);
        double2 disp= make_double2(vA.x-vB.x,vA.y-vB.y);

        while(disp.x < -0.5) disp.x +=1.0;
        while(disp.y < -0.5) disp.y +=1.0;
        while(disp.x > 0.5) disp.x -=1.0;
        while(disp.y > 0.5) disp.y -=1.0;

        Trans(disp,pans);
        };
    };

void periodicBoundaries::move(double2 &p1, const double2 &disp)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    double2 vP;
    p1.x = p1.x+disp.x;
    p1.y = p1.y+disp.y;
    invTrans(p1,vP);
    putInBox(vP);
    Trans(vP,p1);
    };

typedef shared_ptr<periodicBoundaries> PeriodicBoxPtr;

#undef HOSTDEVICE
#endif
