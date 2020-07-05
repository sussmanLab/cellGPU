#ifndef vectorTypeOps_H
#define vectorTypeOps_H
//!component-wise addition of two scalar2s
HOSTDEVICE double2 operator+(const double2 &a, const double2 &b)
    {
    return make_double2(a.x+b.x,a.y+b.y);
    }
//!Less than operator for doubles just sorts by the x-coordinate
HOSTDEVICE bool operator<(const double2 &a, const double2 &b)
    {
    return a.x<b.x;
    }

//!Equality operator tests for.... equality of both elements
HOSTDEVICE bool operator==(const double2 &a, const double2 &b)
    {
    return (a.x==b.x &&a.y==b.y);
    }

//!component-wise subtraction of two double2s
HOSTDEVICE double2 operator-(const double2 &a, const double2 &b)
    {
    return make_double2(a.x-b.x,a.y-b.y);
    }

//!multiplication of double2 by double
HOSTDEVICE double2 operator*(const double &a, const double2 &b)
    {
    return make_double2(a*b.x,a*b.y);
    }


#endif

