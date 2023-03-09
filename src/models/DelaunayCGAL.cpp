#include "DelaunayCGAL.h"
/*! \file DelaunayCGAL.cpp */

/*!
\param V a vector of pairs (Delaunay::Lpoint, particle index)
\param neighs the desired output: a CW-ordered list of the particle indices that neighbor V[0]
*/
bool DelaunayCGAL::LocalTriangulation(const vector<pair<LPoint,int> > &V, vector<int> & neighs)
    {
    neighs.clear();
    int size = V.size();

    Delaunay T;
    T.insert(V.begin(),V.end());

    Delaunay::Face_handle face;
    int li=-1;
    LPoint p=V[0].first;

    face= T.locate(p);

    if(face ==NULL)
        return false;

    if (face->vertex(0)->info()==0)
        li = 0;
    else if (face->vertex(1)->info()==0)
        li = 1;
    else if (face->vertex(2)->info()==0)
        li = 2;
    else
        return false;

    Delaunay::Vertex_handle vh = face->vertex(li);
    Delaunay::Vertex_circulator vc(vh,face);
    int base = vc->info();
    neighs.push_back(base);
    ++vc;
    while(vc->info() != base)
        {
        neighs.push_back(vc->info());
        ++vc;
        };
    cout.flush();

    return true;

    };

/*!
This routine simply calls either the general triangulation routine or the one specialized to square
periodic domains  
The domain is stored as
M= (bxx  bxy
    byx  byy),
and a vector, v, whose components are between zero and one has real position
M*v
*/
void DelaunayCGAL::PeriodicTriangulation(vector<pair<Point,int> > &V, double bxx, double bxy,double byx, double byy)
    {
    if (bxy == 0 && byx == 0 && bxx == byy)
        PeriodicTriangulationSquareDomain(V,bxx,byy);
    else
        PeriodicTriangulationNineSheeted(V,bxx,bxy,byx,byy);
    };

/*!
Perform a periodic triangulation in a non-square domain. This routine constructs the nine-sheeted
covering by hand, computes the triangulation for points in the central domain.
After the routine is called, the class member allneighs will store a list of particle neighbors, each sorted in CW order

\param V A complete set of the points to be triangulated along with their indices
The domain is stored as
M= (bxx  bxy
    byx  byy),
and a vector, v, whose components are between zero and one has real position
M*v
*/
void DelaunayCGAL::PeriodicTriangulationNineSheeted(vector<pair<Point,int> > &V, double bxx, double bxy,double byx, double byy)
    {
    int vnum = V.size();
    allneighs.clear();
    allneighs.resize(vnum);
    //unfortunately, the points have been passed in real coordinates
    double xi11, xi12, xi21,xi22;
    double prefactor = 1.0/(bxx*byy-bxy*byx);
    xi11 = prefactor * byy;
    xi22 = prefactor * bxx;
    xi12 = -prefactor * bxy;
    xi21 = -prefactor * byx;
    vector<double2> virtualCoords(vnum);
    for (int ii = 0; ii < vnum; ++ii)
        {
        virtualCoords[ii].x = xi11*V[ii].first.x() + xi12*V[ii].first.y();
        virtualCoords[ii].y = xi21*V[ii].first.x() + xi22*V[ii].first.y();
        };

    //great, now construct a vector of pairs of LPoints, where the first vnum are the ones in the
    //primary unit cell, and the others are the 8 surrounding unit cells
    vector<pair<LPoint,int> > allPoints(9*vnum);
    vector<double2> primitiveVectors(9);
    primitiveVectors[0] = make_double2(0.0,0.0);
    primitiveVectors[1] = make_double2(-1.,-1);
    primitiveVectors[2] = make_double2(-1.,0.0);
    primitiveVectors[3] = make_double2(-1.0,1.0);
    primitiveVectors[4] = make_double2(0.0,-1.0);
    primitiveVectors[5] = make_double2(0.0,1.0);
    primitiveVectors[6] = make_double2(1.0,-1.0);
    primitiveVectors[7] = make_double2(1.0,0.0);
    primitiveVectors[8] = make_double2(1.0,1.0);

    int index = 0;
    for (int cc = 0; cc < primitiveVectors.size(); ++cc)
        {
        for (int ii = 0; ii < vnum; ++ii)
            {
            double2 point = virtualCoords[ii];
            point.x +=primitiveVectors[cc].x;
            point.y +=primitiveVectors[cc].y;
            double2 realPoint;
            realPoint.x = bxx*point.x + bxy*point.y;
            realPoint.y = byx*point.x + byy*point.y;
            allPoints[index] = make_pair(LPoint(realPoint.x,realPoint.y),index);
            index += 1;
            };
        };
    Delaunay T;
    T.insert(allPoints.begin(),allPoints.end());
    for (int ii = 0; ii < vnum; ++ii)
        {
        vector<int> neighs;
        neighs.reserve(8);

        int li = -1;
        LPoint p=allPoints[ii].first;
        Delaunay::Face_handle face = T.locate(p);
        if (face->vertex(0)->info()==ii)
            li = 0;
        else if (face->vertex(1)->info()==ii)
            li = 1;
        else if (face->vertex(2)->info()==ii)
            li = 2;
    
        Delaunay::Vertex_handle vh = face->vertex(li);
        Delaunay::Vertex_circulator vc(vh,face);
        int base = vc->info();
        neighs.push_back(base % vnum);
        ++vc;
        while(vc->info() != base)
            {
            neighs.push_back((vc->info()) % vnum);
            ++vc;
            };
        allneighs[ii] = neighs;
        };
    };

/*!
Perform a periodic triangulation in a SQUARE domain
\param V A complete set of the points to be triangulated along with their indices
\param boxX the side length of the periodic domain in the x-direction
\parame boxY the side length of the periodic domain in the y-direction

After the routine is called, the class member allneighs will store a list of particle neighbors, each sorted in CW order
*/
void DelaunayCGAL::PeriodicTriangulationSquareDomain(vector<pair<Point,int> > &V, double boxX, double boxY)
    {
    int vnum = V.size();

    Iso_rectangle domain(0.0,0.0,boxX,boxY);
    PDT T(V.begin(),V.end(),domain);

    T.convert_to_1_sheeted_covering();

    int li;
    Locate_type lt;
    vector<Face_handle> fhs(vnum);
    vector<int> lis(vnum);
    vector<bool> located(vnum,false);

    for (int ii = 0; ii < vnum; ++ii)
        {
        if(located[ii]) continue;
        fhs[ii] = T.locate(V[ii].first,lt,lis[ii]);
        int i0,i1,i2;
        i0 = fhs[ii]->vertex(0)->info();
        i1 = fhs[ii]->vertex(1)->info();
        i2 = fhs[ii]->vertex(2)->info();

        if(i0 > ii && !located[i0])
            {
            fhs[i0]=fhs[ii];
            lis[i0]=0;
            located[i0] = true;
            };
        if(i1 > ii && !located[i1])
            {
            fhs[i1]=fhs[ii];
            lis[i1]=1;
            located[i1] = true;
            };
        if(i2 > ii && !located[i2])
            {
            fhs[i2]=fhs[ii];
            lis[i2]=2;
            located[i2] = true;
            };
        };
    allneighs.clear();
    allneighs.resize(vnum);

    vector<int> neighs;
    neighs.reserve(8);
    for (int ii = 0; ii < vnum;++ii)
        {
        neighs.clear();
        Vertex_handle vh = fhs[ii]->vertex(lis[ii]);
        Vertex_circulator vc(vh,fhs[ii]);
        int base = vc->info();
        neighs.push_back(base);

        ++vc;
        while(vc->info() != base)
            {
            neighs.push_back(vc->info());
            ++vc;
            };

        allneighs[ii]=neighs;
        };
    };
