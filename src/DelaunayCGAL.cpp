using namespace std;

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
    int li;
    LPoint p(points[62],points[63]);

    face= T.locate(p);
    if (face->vertex(0)->info()==0) li = 0;
    if (face->vertex(1)->info()==0) li = 1;
    if (face->vertex(2)->info()==0) li = 2;
//    cout << "face handle found...point zero is the " << li << "vertex " << endl;

    Delaunay::Vertex_handle vh = face->vertex(li);
    Delaunay::Vertex_circulator vc(vh,face);
    int base = vc->info();
    //cout << base << endl;cout.flush();
    neighs.push_back(base);
    ++vc;
    while(vc->info() != base)
        {
    //cout << vc->info() << endl;cout.flush();
//        printf("%i\t",vc->info());
        neighs.push_back(vc->info());
        ++vc;
        };

/*
   T.locate(LPoint(points[0],points[1]),lt,li,fh);
    cout << li << endl;cout.flush();

    Delaunay::Vertex_handle vh = fh->vertex(li);
    cout << li << endl;cout.flush();
    Delaunay::Vertex_circulator vc(vh,fh);
    cout << li << endl;cout.flush();
        
    int base = vc->info();
    cout << base << endl;cout.flush();
    neighs.push_back(base);
    ++vc;
    while(vc->info() != base)
        {
    cout << vc->info() << endl;cout.flush();
        printf("%i\t",vc->info());
        neighs.push_back(vc->info());
        ++vc;
        };
    
    */
    
    
   /* 
    for (Delaunay::Finite_faces_iterator fit = T.finite_faces_begin(); fit != T.finite_faces_end(); ++fit)
        {
        Delaunay::Face_handle fh = fit;
        //cout << "T:\t"<< T.triangle(fh) << endl;
//        cout << "v0:\t"<< T.triangle(fh)[0] << endl;
//        cout << "v0idx:\t"<< fh->vertex(0)->info() << endl;
        };
*/



    };

void DelaunayCGAL::PeriodicTriangulation(vector<float> &points, float size)
    {
    vector<Point> V(points.size()/2);
    for (int ii = 0; ii < points.size()/2;++ii)
        {
        V[ii] = Point(points[2*ii],points[2*ii+1]);
        };

    Iso_rectangle domain(0.0,0.0,size,size);
    PDT T(V.begin(),V.end(),domain);

    T.convert_to_1_sheeted_covering();


    Locate_type lt;
    int li;

    for (int ii = 0; ii < points.size()/2; ++ii)
        {
        Face_handle fh = T.locate(V[ii],lt,li);
        Vertex_handle vh = fh->vertex(li);
        vh->info()=ii;


        };
/*
    PDT::Vertex_iterator vit;
    int idx = 0;
    int total_degree = 0;
    for (vit = T.vertices_begin(); vit != T.vertices_end(); ++vit)
        {
        cout << idx << " has degree  " << T.degree(vit) << "  " << vit->point() << "     " << points[2*idx] <<", " << points[2*idx+1] << endl;
        idx +=1;
        total_degree += T.degree(vit);
        };
    cout << total_degree << endl;
*/
    allneighs.clear();
    allneighs.resize(points.size()/2);

    
    for (int ii = 0; ii < points.size()/2;++ii)
        {
        Face_handle fh = T.locate(V[ii],lt,li);
        Vertex_handle vh = fh->vertex(li);
        Vertex_circulator vc(vh,fh);
        vector<int> neighs;
        neighs.reserve(8);
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


/*
        if(vc != 0)
            do {
                ++vc;
                neighs.push_back(vc->info());
                //cout << vc->point() << "   " << vc->info() << endl;
            }while(vc->info()!=base);
  */      

//    cout <<fh->neighbor(0)->index()<< "  " << fh->neighbor(1)->index()<< "  " << fh->neighbor(2)->index() << endl;

    };
