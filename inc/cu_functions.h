#ifndef CUFUNCTIONS_H
#define CUFUNCTIONS_H
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif


HOSTDEVICE void Circumcenter(float2 x1, float2 x2, float2 x3, float2 &xc)
    {
    //Given coordinates (x1,y1),(x2,y2),(x3,y3), feeds the circumcenter to (xc,yc) along with the
    //circumcircle's radius, r.
    //returns false if all three points are on a vertical line
    if(abs(x2.y-x1.y) < EPSILON && abs(x2.y-x3.y) < EPSILON) return;
    float m1,m2,mx1,my1,mx2,my2;

    if(abs(x2.y-x1.y) < EPSILON)
        {
        m2  = -(x3.x-x2.x)/(x3.y-x2.y);
        mx2 = 0.5*(x2.x+x3.x);
        my2 = 0.5*(x2.y+x3.y);
        xc.x  = 0.5*(x2.x+x1.x);
        xc.y  = m2*(xc.x-mx2)+my2;
        }else if(abs(x2.y-x3.y) < EPSILON)
            {
            m1  = -(x2.x-x1.x)/(x2.y-x1.y);
            mx1 = 0.5*(x1.x+x2.x);
            my1 = 0.5*(x1.y+x2.y);
            xc.x  = 0.5*(x3.x+x2.x);
            xc.y  = m1*(xc.x-mx1)+my1;
            }else
            {
            m1  = -(x2.x-x1.x)/(x2.y-x1.y);
            m2  = -(x3.x-x2.x)/(x3.y-x2.y);
            mx1 = 0.5*(x1.x+x2.x);
            mx2 = 0.5*(x2.x+x3.x);
            my1 = 0.5*(x1.y+x2.y);
            my2 = 0.5*(x3.y-x1.y);
            xc.x = (m1*mx1-m2*mx2+my2)/(m1-m2);
            xc.y = m1*(xc.x-mx1)+my1;
            }
    return;
    };

__device__ inline void Circumcircle(float x1, float y1, float x2, float y2, float x3, float y3,
                  float &xc, float &yc, float &r)
    {
    //Given coordinates (x1,y1),(x2,y2),(x3,y3), feeds the circumcenter to (xc,yc) along with the
    //circumcircle's radius, r.
    //returns false if all three points are on a vertical line
    if(abs(y2-y1) < EPSILON && abs(y2-y3) < EPSILON) return;
    float m1,m2,mx1,my1,mx2,my2,dx,dy;

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
    return;
    };

HOSTDEVICE float dot(float2 p1,float2 p2)
    {
    return p1.x*p2.x+p1.y*p2.y;
    };

__device__ inline float norm(float2 p)
    {
    return sqrt(p.x*p.x+p.y*p.y);
    };

__device__ inline int quadrant(float x, float y)
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
            return 1;
        };
    return 2;
    };

//calculate the area of a triangle with a vertex at the origin
HOSTDEVICE float TriangleArea(float2 p1, float2 p2)
    {
    return abs(0.5*(p1.x*p2.y-p1.y*p2.x));
    };

#endif
