#ifndef DELAUNAYCGAL_H
#define DELAUNAYCGAL_H
//uses CGAL to compute periodic or non-periodic delaunay triangulations


#include "std_include.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_2_triangulation_filtered_traits_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <CGAL/Triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <cassert>
#include <fstream>
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Periodic_2_triangulation_filtered_traits_2<K> GT;

typedef CGAL::Periodic_2_triangulation_vertex_base_2<GT>    Vb;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, GT, Vb> VbInfo;

typedef CGAL::Periodic_2_triangulation_face_base_2<GT>      Fb;

typedef CGAL::Triangulation_data_structure_2<VbInfo, Fb>    Tds;
typedef CGAL::Periodic_2_Delaunay_triangulation_2<GT, Tds>  PDT;


typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> NVb;
typedef CGAL::Triangulation_data_structure_2<NVb>           NTds;
typedef CGAL::Delaunay_triangulation_2<K, NTds>  Delaunay;

typedef Delaunay::Point                         LPoint;


typedef PDT::Point             Point;
typedef PDT::Iso_rectangle     Iso_rectangle;
typedef PDT::Vertex_handle     Vertex_handle;
typedef PDT::Locate_type       Locate_type;
typedef PDT::Face_handle       Face_handle;
typedef PDT::Vertex_circulator Vertex_circulator;
/*!
A class for interfacing with the CGAL library. In particular, this lets the user access the
functionality of the 2D periodic and non-periodic schemes for performing a Delaunay Triangulation.
 */
class DelaunayCGAL
    {
    private:
        int N;

    public:
        vector< vector<int> > allneighs; //!<The list of neighbors of every point in the periodic triangulation

        //! Given a vector of points (in the form of pair<PDT::Point p ,int index>), fill the allneighs structure with the neighbor list
        void PeriodicTriangulation(vector<pair<Point,int> > &points,Dscalar size);
        //! given a similar vector of points, calculate the neighbors of the first point in a non-periodic domain.
        bool LocalTriangulation(const vector<pair<LPoint,int> > &points, vector<int> &neighs);

    };



#endif
