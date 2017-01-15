#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "std_include.h"
#include "structures.h"
#include "cu_functions.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/** @defgroup oldFunctions old functions
 * @{
 \brief Utility functions that can be called from host or device
\todo remove any references to functions in this file in favor of a unified set of functions in
cu_functions.h
 */


//!Calculate the circumcenter of (x1,y1),(x2,y2), and the origin...
HOSTDEVICE void CircumCenter(Dscalar x1,Dscalar y1,Dscalar x2,Dscalar y2, Dscalar &xc, Dscalar &yc)
    {
    Dscalar x1norm2,x2norm2,denominator;
    x1norm2 = x1*x1 + y1*y1;
    x2norm2 = x2*x2 + y2*y2;
    denominator = 1/(2.0*Det2x2(x1,y1,x2,y2));

    xc = denominator * Det2x2(x1norm2,y1,x2norm2,y2);
    yc = denominator * Det2x2(x1,x1norm2,x2,x2norm2);
    return;
    };

//!Calculate the circumcenter and radius given 2 points (the third is at the origin)
HOSTDEVICE bool CircumCircle(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2,
                  Dscalar &xc, Dscalar &yc, Dscalar &r)
    {
    CircumCenter(x1,y1,x2,y2,xc,yc);
    Dscalar dx = x1-xc;
    Dscalar dy = y1-yc;
    r = sqrt(dx*dx+dy*dy);
    return true;
    };

//!Calculate the circumcenter and radius for three points on the circumcircle
HOSTDEVICE bool CircumCircle(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2, Dscalar x3, Dscalar y3,
                  Dscalar &xc, Dscalar &yc, Dscalar &r)
    {
    //Given coordinates (x1,y1),(x2,y2),(x3,y3), feeds the circumcenter to (xc,yc) along with the
    //circumcircle's radius, r.
    //returns false if all three points are on a vertical line
    if(abs(y2-y1) < EPSILON && abs(y2-y3) < EPSILON) return false;
    Dscalar m1,m2,mx1,my1,mx2,my2,dx,dy;

    if(abs(y2-y1) < EPSILON)
        {
        m2  = -(x3-x2)/(y3-y2);
        mx2 = 0.5*(x2+x3);
        my2 = 0.5*(y2+y3);
        xc  = 0.5*(x2+x1);
        yc  = m2*(xc-mx2)+my2;
        }else if(abs(y2-y3) < EPSILON)
            {
            m1  = -(x2-x1)/(y2-y1);
            mx1 = 0.5*(x1+x2);
            my1 = 0.5*(y1+y2);
            xc  = 0.5*(x3+x2);
            yc  = m1*(xc-mx1)+my1;
            }else
            {
            m1  = -(x2-x1)/(y2-y1);
            m2  = -(x3-x2)/(y3-y2);
            mx1 = 0.5*(x1+x2);
            mx2 = 0.5*(x2+x3);
            my1 = 0.5*(y1+y2);
            my2 = 0.5*(y3-y1);
            xc = (m1*mx1-m2*mx2+my2)/(m1-m2);
            yc = m1*(xc-mx1)+my1;
            }
    dx = x2-xc;
    dy = y2-yc;
    r = sqrt(dx*dx+dy*dy);
    return true;
    };

//!An overloaded version of the above function when the input/output are Dscalar2's
HOSTDEVICE bool Circumcircle(Dscalar2 &xt, Dscalar2 &x1, Dscalar2 &x2, Dscalar2 &x3,
                  Dscalar2 &xc, Dscalar &rad)
    {
    Dscalar xcen, ycen;
    bool valid = CircumCircle(x1.x,x1.y,x2.x,x2.y,x3.x,x3.y,xcen,ycen,rad);
    if (!valid) return false;
    xc.x=xcen;
    xc.y=ycen;
    Dscalar dx = xt.x-xcen;
    Dscalar dy = xt.y-ycen;
    Dscalar drsqr = sqrt(dx*dx+dy*dy);

    return ((drsqr <=rad)? true: false);
    };

//!Given a separation and a scale, returns true if (xc, yc) is on the same half plane with normal \vec{sep} and passing through midScale*\vec{sep} as the origin
HOSTDEVICE bool halfPlane(Dscalar sepx, Dscalar sepy, Dscalar midScale, Dscalar xc, Dscalar yc)
    {
    Dscalar norm = sqrt(sepx*sepx+sepy*sepy);
    Dscalar normx = sepx/norm;
    Dscalar normy = sepy/norm;
    Dscalar midx = midScale*sepx;
    Dscalar midy = midScale*sepy;
    Dscalar dd = normx*midx+normy*midy;

    int cSign = -1;
    int psign = 1;
    if ((xc*normx+yc*normy-dd) <0) psign = -1;

    return (psign==cSign);
    };

//!Calculates the area of a triangle with one vertex at the origin
HOSTDEVICE Dscalar TriangleArea(Dscalar x1, Dscalar y1, Dscalar x2, Dscalar y2)
    {
    return 0.5*(x1*y2-x2*y1);
    };

//!Calculate which quadrant the point (x,y) is in
HOSTDEVICE int Quadrant(Dscalar x, Dscalar y)
    {
    if(x>=0)
        {
        if (y >=0)
            {return 0;
                }else{
                return  3;
                };
        }else{
        if (y>=0)
            {return 1;
                }else{
                return  2;
                };
        };
    return -1;
    };

/** @} */ //end of group declaration
#undef HOSTDEVICE

#endif


