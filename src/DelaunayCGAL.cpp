using namespace std;

#include "DelaunayCGAL.h"



void DelaunayCGAL::Triangulate(vector<float> &points, float size)
    {
    vector<Point> V(points.size()/2);
    for (int ii = 0; ii < points.size()/2;++ii)
        {
        V[ii] = Point(points[2*ii],points[2*ii+1]);
        };

    Iso_rectangle domain(0.0,0.0,size,size);
    PDT T(V.begin(),V.end(),domain);

    Locate_type lt;
    int li;
    Face_handle fh = T.locate(V[21],lt,li);
    Vertex_handle vh = fh->vertex( (li + 1) % 3 );
    Vertex_circulator vc(vh,fh);

    if(vc != 0)
        do {
            cout << vc->point() << endl;
        }while(true);
//    cout <<fh->neighbor(0)->index()<< "  " << fh->neighbor(1)->index()<< "  " << fh->neighbor(2)->index() << endl;

    };
