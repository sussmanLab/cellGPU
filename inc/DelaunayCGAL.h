#ifndef DELAUNAYCGAL_H
#define DELAUNAYCGAL_H
//uses CGAL to compute periodic or non-periodic delaunay triangulations


using namespace std;
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_2_triangulation_traits_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <cassert>
#include <fstream>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Periodic_2_triangulation_traits_2<K> GT;
typedef CGAL::Periodic_2_Delaunay_triangulation_2<GT> PDT;

typedef PDT::Point             Point;
typedef PDT::Iso_rectangle     Iso_rectangle;
typedef PDT::Vertex_handle     Vertex_handle;
typedef PDT::Locate_type       Locate_type;
typedef PDT::Face_handle       Face_handle; 

typedef PDT::Vertex_circulator Vertex_circulator;

class DelaunayCGAL
    {
    private:
        int N;

    public:
        void Triangulate(vector<float> &points,float size);


    };



#endif
