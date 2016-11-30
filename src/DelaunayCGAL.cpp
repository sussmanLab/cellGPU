using namespace std;

#include "cuda_runtime.h"
#include "vector_types.h"

#include "DelaunayCGAL.h"

void DelaunayCGAL::LocalTriangulation(vector<pair<LPoint,int> > &V, vector<int> & neighs)
    {
    neighs.clear();
    int size = V.size();
    /*
    int size = points.size()/2;
    //vector<LPoint> V(size);
    vector<pair<LPoint,int> > V(size);
    float max = 0.0;
    for (int ii = 0; ii < size;++ii)
        {
        float valx = points[2*ii];
        float valy = points[2*ii+1];
    //    if (fabs(valx)> max)
    //        max = fabs(valx);
    //    if (fabs(valy)> max)
    //        max = fabs(valy);
        V[ii] = make_pair(LPoint(valx,valy),ii);
        };
    */



    Delaunay T;
    T.insert(V.begin(),V.end());

    Delaunay::Face_handle face;
    int li=-1;
    LPoint p(points[0],points[1]);

    face= T.locate(p);
    if (face->vertex(0)->info()==0) li = 0;
    if (face->vertex(1)->info()==0) li = 1;
    if (face->vertex(2)->info()==0) li = 2;

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

    };

//void DelaunayCGAL::PeriodicTriangulation(vector<Point> &V, float size)
void DelaunayCGAL::PeriodicTriangulation(vector<pair<Point,int> > &V, float size)
    {
    int vnum = V.size();

    Iso_rectangle domain(0.0,0.0,size,size);
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
        //Face_handle fh = T.locate(V[ii],lts[ii],li);
        //Face_handle fh = T.locate(V[ii],lt,li);
        Vertex_handle vh = fhs[ii]->vertex(lis[ii]);
        //Vertex_handle vh = fh->vertex(li);
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
