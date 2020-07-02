#ifndef DELAUNAYCGAL_H
#define DELAUNAYCGAL_H

#include "vector_types.h"
#include "vector_functions.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Periodic_2_triangulation_face_base_2.h>
#include <CGAL/Periodic_2_triangulation_vertex_base_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
/*! \file DelaunayCGAL.h */
//provides an interface to periodic and non-periodic 2D Delaunay triangulations via the CGAL library
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Periodic_2_Delaunay_triangulation_traits_2<K>             Gt;

typedef CGAL::Periodic_2_triangulation_vertex_base_2<Gt>                Vbb;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned, Gt, Vbb>  Vb;
typedef CGAL::Periodic_2_triangulation_face_base_2<Gt>                  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                    Tds;
typedef CGAL::Periodic_2_Delaunay_triangulation_2<Gt, Tds>              PDT;



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
//! Access the 2D periodic and non-periodic functionality of CGAL Delaunay triangulations
/*!
A class for interfacing with the CGAL library.
In particular, this lets the user access the functionality of the 2D periodic and non-periodic
schemes for performing a Delaunay Triangulation.
A public member variable maintains a convenient data structure for keeping track of the most recently
performed complete triangulation of a periodic point set.
 */
class DelaunayCGAL
    {
    public:
        vector< vector<int> > allneighs; //!<The list of neighbors of every point in the periodic triangulation

        //! Given a vector of points (in the form of pair<PDT::Point p ,int index>), fill the allneighs structure with the neighbor list. Calls one of the routines below
        void PeriodicTriangulation(vector<pair<Point,int> > &points,double bxx, double bxy, double byx, double byy);
        //! Given a vector of points (in the form of pair<PDT::Point p ,int index>), explicitly constructing the covering and using CGAL's non-periodic routines
        void PeriodicTriangulationNineSheeted(vector<pair<Point,int> > &points,double bxx, double bxy, double byx, double byy);
        //! Given a vector of points (in the form of pair<PDT::Point p ,int index>), fill the allneighs structure with the neighbor list
        void PeriodicTriangulationSquareDomain(vector<pair<Point,int> > &points,double boxX, double boxY);
        //! given a similar vector of points, calculate the neighbors of the first point in a non-periodic domain.
        bool LocalTriangulation(const vector<pair<LPoint,int> > &points, vector<int> &neighs);
    };
#endif
