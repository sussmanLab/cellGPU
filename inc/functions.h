#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "structures.h"

using namespace std;

namespace voroguppy
{
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

HOSTDEVICE bool Circumcircle(dbl x1, dbl y1, dbl x2, dbl y2, dbl x3, dbl y3,
                  dbl &xc, dbl &yc, dbl &r)
    {
    //Given coordinates (x1,y1),(x2,y2),(x3,y3), feeds the circumcenter to (xc,yc) along with the
    //circumcircle's radius, r.
    //returns false if all three points are on a vertical line
    if(abs(y2-y1) < EPSILON && abs(y2-y3) < EPSILON) return false;
    dbl m1,m2,mx1,my1,mx2,my2,dx,dy;

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

HOSTDEVICE bool Circumcircle(pt &xt, pt &x1, pt &x2, pt &x3,
                  pt &xc, dbl &rad)
    {
    //overloaded version when the input/output are pt objects

    dbl xcen, ycen;
    bool valid = Circumcircle(x1.x,x1.y,x2.x,x2.y,x3.x,x3.y,xcen,ycen,rad);
    if (!valid) return false;
    xc.x=xcen;
    xc.y=ycen;
    dbl dx = xt.x-xcen;
    dbl dy = xt.y-ycen;
    dbl drsqr = sqrt(dx*dx+dy*dy);

    return ((drsqr <=rad)? true: false);
    };

//given a separation and a scale, returns true if (xc, yc) is on the same half plane with normal \vec{sep} and passing through midScale*\vec{sep} as the origin
HOSTDEVICE bool halfPlane(dbl sepx, dbl sepy, dbl midScale, dbl xc, dbl yc)
    {
    dbl norm = sqrt(sepx*sepx+sepy*sepy);
    dbl normx = sepx/norm;
    dbl normy = sepy/norm;
    dbl midx = midScale*sepx;
    dbl midy = midScale*sepy;
    dbl dd = normx*midx+normy*midy;

    int cSign = -1;
    int psign = 1;
    if ((xc*normx+yc*normy-dd) <0) psign = -1;

    return (psign==cSign);
    };

//calculates the area of a triangle with one vertex at the origin
HOSTDEVICE dbl TriangleArea(dbl x1, dbl y1, dbl x2, dbl y2)
    {
    return 0.5*(x1*y2-x2*y1);
    };

HOSTDEVICE int quadrant(dbl x, dbl y)
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




// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE

};
#endif


