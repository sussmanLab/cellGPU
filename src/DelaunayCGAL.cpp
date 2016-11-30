using namespace std;

#include "cuda_runtime.h"
#include "vector_types.h"

#include "DelaunayCGAL.h"

void DelaunayCGAL::LocalTriangulation(vector<float> &points, vector<int> & neighs)
    {
    neighs.clear();
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




    Delaunay T;
    T.insert(V.begin(),V.end());

    Delaunay::Face_handle face;
    int li=-1;
    LPoint p(points[0],points[0]);

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

void DelaunayCGAL::PeriodicTriangulation(vector<Point> &V, float size)
    {
    int vnum = V.size();

    Iso_rectangle domain(0.0,0.0,size,size);
    PDT T(V.begin(),V.end(),domain);

    T.convert_to_1_sheeted_covering();


    int li;
    Locate_type lt;
    vector<Face_handle> fhs(vnum);
    vector<int> lis(vnum);
    for (int ii = 0; ii < vnum; ++ii)
        {
        //Face_handle fh = T.locate(V[ii],lt,li);
        fhs[ii] = T.locate(V[ii],lt,lis[ii]);
        //Vertex_handle vh = fhs[ii]->vertex(li);
        Vertex_handle vh = fhs[ii]->vertex(lis[ii]);
        vh->info()=ii;


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
