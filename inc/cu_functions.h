#ifndef CUFUNCTIONS_H
#define CUFUNCTIONS_H


#include "std_include.h"
#include "Matrix.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//!Calculate the determinant of a 2x2 matrix
HOSTDEVICE Dscalar Det2x2(const Dscalar &x11,const Dscalar &x12, const Dscalar &x21, const Dscalar &x22)
    {
    return x11*x22-x12*x21;
    };

//!Calculates the circumcenter of the circle passing through x1, x2 and the origin
HOSTDEVICE void Circumcenter(const Dscalar2 &x1, const Dscalar2 &x2, Dscalar2 &xc)
    {
    Dscalar x1norm2,x2norm2,denominator;
    x1norm2 = x1.x*x1.x + x1.y*x1.y;
    x2norm2 = x2.x*x2.x + x2.y*x2.y;
    denominator = 0.5/(Det2x2(x1.x,x1.y,x2.x,x2.y));

    xc.x = denominator * Det2x2(x1norm2,x1.y,x2norm2,x2.y);
    xc.y = denominator * Det2x2(x1.x,x1norm2,x2.x,x2norm2);
    };

//!Calculates the circumcenter through x1,x2,x3
HOSTDEVICE void Circumcenter(const Dscalar2 &x1, const Dscalar2 &x2, const Dscalar2 &x3, Dscalar2 &xc)
    {
    Dscalar amcx,amcy,bmcx,bmcy,amcnorm2,bmcnorm2,denominator;

    amcx=x1.x-x3.x;
    amcy=x1.y-x3.y;
    amcnorm2 = amcx*amcx+amcy*amcy;
    bmcx=x2.x-x3.x;
    bmcy=x2.y-x3.y;
    bmcnorm2 = bmcx*bmcx+bmcy*bmcy;

    denominator = 0.5/(Det2x2(amcx,amcy,bmcx,bmcy));

    xc.x = x3.x + denominator * Det2x2(amcnorm2,amcy,bmcnorm2,bmcy);
    xc.y = x3.y + denominator * Det2x2(amcx,amcnorm2,bmcx,bmcnorm2);

    };

//!Get the circumcenter and radius, given one of the points on the circumcircle is the origin...
HOSTDEVICE void Circumcircle(const Dscalar2 &x1, const Dscalar2 &x2, Dscalar2 &xc, Dscalar &radius)
    {
    Circumcenter(x1,x2,xc);
    Dscalar dx = (x1.x-xc.x);
    Dscalar dy = (x1.y-xc.y);
    radius = sqrt(dx*dx+dy*dy);
    };

//!Get the circumcenter and radius given three distinct points on the circumcirle
HOSTDEVICE void Circumcircle(const Dscalar2 &x1, const Dscalar2 &x2, const Dscalar2 &x3, Dscalar2 &xc, Dscalar &radius)
    {
    Circumcenter(x1,x2,x3,xc);
    Dscalar dx = (x1.x-xc.x);
    Dscalar dy = (x1.y-xc.y);
    radius = sqrt(dx*dx+dy*dy);
    };

//!The dot product between two vectors of length two.
HOSTDEVICE Dscalar dot(const Dscalar2 &p1, const Dscalar2 &p2)
    {
    return p1.x*p2.x+p1.y*p2.y;
    };


//!Calculate the area of a triangle with a vertex at the origin
HOSTDEVICE Dscalar TriangleArea(const Dscalar2 &p1, const Dscalar2 &p2)
    {
    return abs(0.5*(p1.x*p2.y-p1.y*p2.x));
    };

//!Calculate matrix of derivatives needed in the 2D SPV model... this is the change in a voronoi vertex given a change of a Delaunay vertex at the origin, given that rij and rik tell you where the other two Delaunay vertices are.
HOSTDEVICE void getdhdr(Matrix2x2 &dhdr,const Dscalar2 &rij,const Dscalar2 &rik)
    {
    Matrix2x2 Id;
    dhdr=Id;

    Dscalar2 rjk;
    rjk.x =rik.x-rij.x;
    rjk.y =rik.y-rij.y;

    Dscalar rikDotrik=dot(rik,rik);
    Dscalar rikDotrjk=dot(rik,rjk);
    Dscalar rijDotrjk=dot(rij,rjk);
    Dscalar rij2=dot(rij,rij);


    Dscalar2 dDdriOD,z;
    Dscalar cpi = 1.0/(rij.x*rjk.y - rij.y*rjk.x);
    //"D" is really 2*cp*cp
    Dscalar D = 0.5*cpi*cpi;
    //betaD has an extra D for computational efficiency
    //same with gammaD
    Dscalar betaD = -D*rikDotrik*rijDotrjk;
    Dscalar gammaD = D*rij2*rikDotrjk;

    z.x = betaD*rij.x+gammaD*rik.x;
    z.y = betaD*rij.y+gammaD*rik.y;

    dDdriOD.x = (-2.0*rjk.y)*cpi;
    dDdriOD.y = (2.0*rjk.x)*cpi;

    dhdr -= ((betaD+gammaD)*Id+dyad(z,dDdriOD));

    //reuse dDdriOd, but here as dbDdri
    dDdriOD.x = D*(2.0*rijDotrjk*rik.x+rikDotrik*rjk.x);
    dDdriOD.y = D*(2.0*rijDotrjk*rik.y+rikDotrik*rjk.y);

    dhdr += dyad(rij,dDdriOD);

    //reuse dDdriOd, but here as dgDdri
    dDdriOD.x = D*(-2.0*rikDotrjk*rij.x-rij2*rjk.x);
    dDdriOD.y = D*(-2.0*rikDotrjk*rij.y-rij2*rjk.y);

    dhdr += dyad(rik,dDdriOD);
    //dhdr = Id+D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));

    return;
    };


/*! Given three consecutive voronoi vertices and some cell information, compute -dE/dv
 Adiff = KA*(A_i-A_0)
 Pdiff = KP*(P_i-P_0)
 */
HOSTDEVICE void computeForceSetAVM(const Dscalar2 &vcur, const Dscalar2 &vlast, const Dscalar2 &vnext,
                                   const Dscalar &Adiff, const Dscalar &Pdiff,
                                   Dscalar2 &dEdv)
    {
    Dscalar2 dlast,dnext,dAdv,dPdv;

    dAdv.x = 0.5*(vlast.y-vnext.y);
    dAdv.y = 0.5*(vlast.x-vnext.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y = vlast.y-vcur.y;
    Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

    dEdv.x = 2.0*Adiff*dAdv.x + 2.0*Pdiff*dPdv.x;
    dEdv.y = 2.0*Adiff*dAdv.y + 2.0*Pdiff*dPdv.y;
    }


#ifdef ENABLE_CUDA
#include "cuda_runtime.h"
//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        cout << "Requested GPU (device " << USE_GPU<<") does not exist. Stopping triangulation" << endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)    cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        if (verbose)
            {
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        };
    if (!verbose)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    };
#endif

#endif
