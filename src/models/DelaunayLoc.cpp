#define ENABLE_CUDA

#include "DelaunayLoc.h"
#include "DelaunayCGAL.h"

/*! \file DelaunayLoc.cpp */

/*!
\param points references to a vector of new location for points, formatted as {x1,y1,x2,y2,...}
*/
void DelaunayLoc::setPoints(vector<Dscalar> &points)
    {
    nV=points.size()/2;
    pts.resize(nV);
    triangulated = false;
    cellsize = 2.0;
    for (unsigned int ii = 0; ii<nV; ++ii)
        {
        Dscalar2 point;
        point.x=points[ii*2]; point.y=points[ii*2+1];
        pts[ii]=point;
        };
    };
/*!
\param points ArrayHandle of Dscalar2's that has already accessed a GPUArray containing the
new desired points
\param N the number of cells that the ArrayHandle knows about
*/
void DelaunayLoc::setPoints(ArrayHandle<Dscalar2> &points, int N)
    {
    nV=N;
    pts.clear();
    pts.reserve(nV);
    triangulated = false;
    cellsize = 2.0;
    for (int ii = 0; ii < nV; ++ii)
        {
        pts.push_back(points.data[ii]);
        };
    };

/*!
\param points a GPUArray of Dscalar2's with the new desired points
*/
void DelaunayLoc::setPoints(GPUArray<Dscalar2> &points)
    {
    nV=points.getNumElements();
    //pts.clear();
    pts.resize(nV);
    triangulated = false;
    cellsize = 2.0;
    ArrayHandle<Dscalar2> hp(points,access_location::host,access_mode::read);
    for (int ii = 0; ii < nV; ++ii)
        {
        pts[ii].x=hp.data[ii].x;
        pts[ii].y=hp.data[ii].y;
        };
    };

/*!
\param points a vector of Dscalar2's with the new desired points
*/
void DelaunayLoc::setPoints(vector<Dscalar2> &points)
    {
    nV=points.size();
    pts.clear();pts.reserve(nV);
    for (int ii = 0; ii < nV; ++ii)
        {
        pts.push_back(points[ii]);
        };
    triangulated = false;
    cellsize = 2.0;
    };

/*!
\param bx a gpubox that the DelaunayLoc object should use in internal computations
*/
void DelaunayLoc::setBox(gpubox &bx)
    {
    Box = make_shared<gpubox>();
    Dscalar b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    if (bx.isBoxSquare())
        Box->setSquare(b11,b22);
    else
        Box->setGeneral(b11,b12,b21,b22);
    };
/*!
\param csize the size of the grid boxes to use for the internal cell list
\pre the internal box is already set, and the points are already set
\post points are sorted into cells
*/
void DelaunayLoc::initialize(Dscalar csize)
    {
    cellsize = csize;
    cList.setNp(nV);
    cList.setBox(Box);
    cList.setGridSize(cellsize);
    cList.setParticles(pts);
    cList.compute();
    };

/*!
This routine uses the cell list to look for particles that form a quadrilateral surrounding cell i.
The search proceeds by looking in concentric shells of nearby cells until a particle from every
quadrant around cell i is found.
\param i the index of the cell in question
\param P0 a reference to the indices of cells that form the enclosing polygon
\param P1 a reference to the positions of the cells forming the enclosing polygon relative to cell i
*/
void DelaunayLoc::getPolygon(int i, vector<int> &P0,vector<Dscalar2> &P1)
    {
    vector<int> Pt(4,-1);
    Dscalar2 np;
    np.x=-1;np.y=-1;
    vector<Dscalar2> Pt2(4,np);
    P0.resize(4);
    P1.resize(4);

    vector<Dscalar> dists(4,1e6);
    Dscalar2 v = pts[i];
    int cidx = cList.positionToCellIndex(v.x,v.y);
    vector<bool> found(4,false);
    int wmax = cList.getXsize();
    //while a data point in a quadrant hasn't been found, expand the size of the search grid and keep looking
    int width = 0;
    vector<int> cellneighs;cellneighs.reserve(25);
    int idx;
    Dscalar2 disp;
    Dscalar nrm;
    ArrayHandle<unsigned int> h_cs(cList.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> h_idx(cList.idxs,access_location::host,access_mode::read);
    while(!found[0]||!found[1]||!found[2]||!found[3])
        {
        cList.getCellShellNeighbors(cidx,width,cellneighs);
        for (int cc = 0; cc < cellneighs.size(); ++cc)
            {
            int numberInCell = h_cs.data[cellneighs[cc]];
            for (int pp = 0; pp < numberInCell;++pp)
                {
                idx = h_idx.data[cList.cell_list_indexer(pp,cellneighs[cc])];
                if (idx == i ) continue;
                Box->minDist(pts[idx],v,disp);
                nrm = sqrt(disp.x*disp.x+disp.y*disp.y);
                int q = Quadrant(disp.x,disp.y);
                if(!found[q]||nrm < dists[q])
                    {
                    found[q]=true;
                    dists[q]=nrm;
                    P1[q]=disp;
                    P0[q]=idx;
                    };
                };
            };

        width +=1;
        if (width >= wmax) return;
        };//end loop over cells
    };

/*!
The main workhorse of the class. First find a set of cells that forms a polygon around cell i, then
find all particles in the circumcircles formed by cell i and any two consecutive members of the
polygon
\param i the cell to get the candidate 1-ring of
\param DTringIdx a reference to a vector of cell indices that will make up the candidate 1-ring
\param DTring a reference to a vector of relative cell positions that will make up the candidate 1-ring
\post DTringIdx and DTring contain cells, of which the Delaunay neighbors of cell i are a strict
subset
*/
void DelaunayLoc::getOneRingCandidate(int i, vector<int> &DTringIdx, vector<Dscalar2> &DTring)
    {
    //first, find a polygon enclosing vertex i
    vector<int> P0;//index of vertices forming surrounding sqaure
    vector<Dscalar2> P1;//relative position of vertices forming surrounding square
    getPolygon(i,P0,P1);

    //now, get the cells in the circumcircles
    Dscalar2 v;
    v.x=pts[i].x;v.y=pts[i].y;
    DTring.clear();
    DTringIdx.clear();

    int reduceSize = 30;
    DTring.reserve(2*reduceSize); DTringIdx.reserve(2*reduceSize);
    Dscalar2 vc;
    vc.x=0.0; vc.y=0.0;
    DTring.push_back(vc);
    DTringIdx.push_back(i);
    for (int jj = 0; jj < 4;++jj)
        {
        DTringIdx.push_back(P0[jj]);
        DTring.push_back(P1[jj]);
        };

    vector<Dscalar2> Q0;//vector of circumcenters formed by vertex i and the P_i
    Dscalar2 Qnew;
    Qnew.x=0.0;Qnew.y=0.0;
    bool valid;
    vector<Dscalar> rads;
    Q0.reserve(4);
    rads.reserve(4);
    Dscalar radius;
    Dscalar vx = 0.0;Dscalar vy = 0.0;
    for (int ii = 0; ii < 4; ++ii)
        {
        Circumcircle(P1[ii],P1[(ii+1)%4],Qnew,radius);
        Q0.push_back(Qnew);
        rads.push_back(radius*1.0001);
        };


    vector<int> cellns;
    cellns.reserve(100);//often 16*4 or 25*4
    vector<int> cns;
    int idx;
    for (int ii = 0; ii < 4; ++ii)
        {
        int cix = cList.positionToCellIndex(v.x+Q0[ii].x,v.y+Q0[ii].y);

        int wcheck = ceil(rads[ii]/cList.getBoxsize())+1;
        cList.getCellNeighbors(cix,wcheck,cns);
        //cellschecked += cns.size();
        for (int cc = 0; cc < cns.size(); ++cc)
            cellns.push_back(cns[cc]);
        };
    //only look in each cell once
    sort(cellns.begin(),cellns.end());
    cellns.erase(unique(cellns.begin(),cellns.end() ), cellns.end() );
    //cellschecked = cellns.size();

    Dscalar2 tocenter;
    Dscalar2 disp;
    bool repeat=false;
    Dscalar rr;
    ArrayHandle<unsigned int> h_cs(cList.cell_sizes,access_location::host,access_mode::read);
    ArrayHandle<int> h_idx(cList.idxs,access_location::host,access_mode::read);
    for (int cc = 0; cc < cellns.size(); ++cc)
        {
        int numberInCell = h_cs.data[cellns[cc]];
        for (int pp = 0; pp < numberInCell;++pp)
            {
            idx = h_idx.data[cList.cell_list_indexer(pp,cellns[cc])];
            //exclude anything already in the ring (vertex and polygon)
            if (idx == i || idx == DTringIdx[1] || idx == DTringIdx[2] ||
                            idx == DTringIdx[3] || idx == DTringIdx[4]) continue;
            Box->minDist(pts[idx],v,disp);
            //how far is the point from the circumcircle's center?
            repeat = false;
            for (int qq = 0; qq < 4; ++qq)
                {
                if (repeat) continue;
                rr=rads[qq];
                rr = rr*rr;
                Box->minDist(disp,Q0[qq],tocenter);
                if(tocenter.x*tocenter.x+tocenter.y*tocenter.y<rr)
                    {
                    //the point is in at least one circumcircle...
                     repeat = true;
                     DTringIdx.push_back(idx);
                     DTring.push_back(disp);
                    };

                };
            };
        };
    //if the candidate 1-ring is very large, see if a simple algorithm can reduce it
    if (DTring.size() > reduceSize)
        {
        reduceOneRing(i,DTringIdx,DTring);
        };
    //candidates = DTring.size();
    };

/*!
A simple algorithm to try to reduce the size of the candidate 1-ring. This exploits the fact that
getPolygon doesn't try very hard to find a "good" enclosing polygon for cell i, choosing a
different set of cells to form the polygon might lead to a smaller candidate 1-ring. This then
reduces the computational cost of going from the candidate 1-ring to the true set of Delaunay
neighbors. This matters a lot for analyzing random point sets, but if the cells are pretty regular
this routine will almost never be called.
\param i the cell to get the candidate 1-ring of
\param DTringIdx a reference to a vector of cell indices that will make up the candidate 1-ring
\param DTring a reference to a vector of relative cell positions that will make up the candidate 1-ring
subset
*/
void DelaunayLoc::reduceOneRing(int i, vector<int> &DTringIdx, vector<Dscalar2> &DTring)
    {
    //basically, see if an enclosing polygon with a smaller sum of circumcircle radii can be found
    //start with the vertex i
    vector<int> newRingIdx; newRingIdx.reserve(50);
    vector<Dscalar2> newRing;     newRing.reserve(50);
    Dscalar2 v;
    v.x=0.0;v.y=0.0;
    newRing.push_back(v);
    newRingIdx.push_back(i);

    vector<Dscalar2> Q0;//vector of circumcenters formed by vertex i and the P_i
    Dscalar2 Qnew,Qnew2;
    Qnew.x=0.0;Qnew.y=0.0;Qnew2.x=0.0;Qnew2.y=0.0;
    bool valid;
    vector<int> P0(4);
    vector<Dscalar2> P1(4);
    P0[0]=DTringIdx[1];
    P0[1]=DTringIdx[2];
    P0[2]=DTringIdx[3];
    P0[3]=DTringIdx[4];
    P1[0]=DTring[1];
    P1[1]=DTring[2];
    P1[2]=DTring[3];
    P1[3]=DTring[4];

    Dscalar radius;
    Dscalar vx = 0.0;Dscalar vy = 0.0;

    int Psize=P1.size();
    vector<Dscalar> rads;
    Q0.reserve(4);
    rads.reserve(4);
    for (int ii = 0; ii < Psize; ++ii)
        {
        Circumcircle(P1[ii],P1[(ii+1)%Psize],Qnew,radius);
        Q0.push_back(Qnew);
        rads.push_back(radius);
        };

    for (int nn = 5; nn < DTring.size(); ++nn)
        {
        int q =Quadrant(DTring[nn].x,DTring[nn].y);
        int polyi1 = (q+1)%Psize;
        int polyi2 = q-1;
        if(polyi2 < 0) polyi2 = Psize -1;
        Dscalar r1,r2;
        Circumcircle(DTring[nn],P1[polyi1],Qnew,r1);
        Circumcircle(P1[polyi2],DTring[nn],Qnew2,r2);
        if(r1+r2 < rads[q]+rads[polyi2])
            {
            P1[q]=DTring[nn];
            Q0[q]=Qnew;
            Q0[polyi2]=Qnew2;
            rads[q]=r1;
            rads[polyi2]=r2;
            P0[q]=DTringIdx[nn];
            };
        };

    bool repeat = false;
    Dscalar2 tocenter;
    Dscalar rr;
    for (int pp = 1; pp < DTring.size(); ++pp)
        {
        //check if DTring[pp] is in any of the new circumcircles
        repeat = false;
        for (int qq = 0; qq < Q0.size(); ++qq)
            {
            if (repeat) continue;
            rr=rads[qq]*1.0001;
            rr = rr*rr;
            Box->minDist(DTring[pp],Q0[qq],tocenter);
            if(tocenter.x*tocenter.x+tocenter.y*tocenter.y<rr)
                {
                newRing.push_back(DTring[pp]);
                newRingIdx.push_back(DTringIdx[pp]);
                repeat = true;
                };
            };

        };

    DTring.swap(newRing);
    DTringIdx.swap(newRingIdx);
    //candidates = DTring.size();
    };
/*!
call the CGAL library (for non-periodic 2D Delaunay triangulations) through the DelaunayCGAL class
to go from the candidate 1-ring of cell i to its true set of delaunay neighbors.
\param i the cell in question
\param neighbors a reference to a vector of cell indices that are the Delaunay neighbors of i
\post neighbors is sorted so that the indices refer to the Delaunay neighbors in CCW order
*/
bool DelaunayLoc::getNeighborsCGAL(int i, vector<int> &neighbors)
    {
    //first, get candidate 1-ring
    getOneRingCandidate(i,DTringIdxCGAL,DTringCGAL);

    //call another algorithm to triangulate the candidate set
    DelaunayCGAL delcgal;

    vector<pair<LPoint,int> > Pnts(DTringCGAL.size());
    for (int ii = 0; ii < DTringCGAL.size(); ++ii)
        {
        Pnts[ii] = make_pair(LPoint(DTringCGAL[ii].x,DTringCGAL[ii].y),ii);
        };
    bool success = delcgal.LocalTriangulation(Pnts, neighbors);

    for (int nn = 0; nn < neighbors.size(); ++nn)
        neighbors[nn] = DTringIdxCGAL[neighbors[nn]];

    if (success)
        return true;
    else
        return false;
    };


/*!
If CGAL is unavailable, call the DelaunayNP class to go from the candidate 1-ring of cell i to
its true set of delaunay neighbors.
\param i the cell in question
\param neighbors a reference to a vector of cell indices that are the Delaunay neighbors of i
\post neighbors is sorted so that the indices refer to the Delaunay neighbors in CCW order
*/
void DelaunayLoc::getNeighbors(int i, vector<int> &neighbors)
    {
    DelaunayCell DCell;
    //first, get candidate 1-ring
    vector<int> DTringIdx;
    vector<Dscalar2> DTring;
    getOneRingCandidate(i,DTringIdx,DTring);


    //call another algorithm to triangulate the candidate set
    DelaunayNP del(DTring);
    del.triangulate();


    //pick out the triangulation of the desired vertex
    int sv = del.mapi[0];

    //get the Delaunay neighbors of that point
    del.DT.getNeighbors(sv,neighbors);
    DCell.setSize(neighbors.size());
    Dscalar2 pi;
    for (int ii = 0; ii < DCell.n; ++ii)
        {
        del.getSortedPoint(neighbors[ii],pi);
        DCell.Dneighs[ii]=pi;
        };

    //calculate the cell geometric properties, and put the points in CW order
    DCell.getCW();

    //convert neighbors to global indices,
    //and store the neighbor indexes in clockwise order
    vector<int> nidx;nidx.reserve(neighbors.size());
    for (int nn = 0; nn < neighbors.size(); ++nn)
        {
        int localidx = del.deSortPoint(neighbors[DCell.CWorder[nn].second]);
        nidx.push_back(DTringIdx[localidx]);
        };
    for (int nn = 0; nn < neighbors.size(); ++nn)
        {
        neighbors[nn] = nidx[nn];
        };
    };

/*!
Similar to getNeighbors above, this uses the DelaunayNP class to ge the Delaunay neighbors of cell i,
and also calculates geometric properties of the Voronoi cell that results. Useful for debugging and
testing.
its true set of delaunay neighbors.
\param i the cell in question
\param neighbors a reference to a vector of cell indices that are the Delaunay neighbors of i
\param DCell a reference to the Voronoi cell of point i, with some geometric propertie of it calculated.
\param timing keep track of some timing info if true
\post neighbors is sorted so that the indices refer to the Delaunay neighbors in CCW order
*/

void DelaunayLoc::triangulatePoint(int i, vector<int> &neighbors, DelaunayCell &DCell,bool timing)
    {
    //first, get candidate 1-ring
    vector<int> DTringIdx;
    vector<Dscalar2> DTring;
    getOneRingCandidate(i,DTringIdx,DTring);


    //call another algorithm to triangulate the candidate set
    DelaunayNP del(DTring);
    del.triangulate();

    //pick out the triangulation of the desired vertex
    int sv = del.mapi[0];

    //get the Delaunay neighbors of that point
    del.DT.getNeighbors(sv,neighbors);
    DCell.setSize(neighbors.size());
    Dscalar2 pi;
    for (int ii = 0; ii < DCell.n; ++ii)
        {
        del.getSortedPoint(neighbors[ii],pi);
        DCell.Dneighs[ii]=pi;
        };

    //calculate the cell geometric properties, and put the points in CW order
    DCell.Calculate();

    //convert neighbors to global indices,
    //and store the neighbor indexes in clockwise order
    vector<int> nidx;nidx.reserve(neighbors.size());
    for (int nn = 0; nn < neighbors.size(); ++nn)
        {
        int localidx = del.deSortPoint(neighbors[DCell.CWorder[nn].second]);
        nidx.push_back(DTringIdx[localidx]);
        };
    for (int nn = 0; nn < neighbors.size(); ++nn)
        {
        neighbors[nn] = nidx[nn];
        };
    };

/*!
Given a cell index and a propsed set of neighbors in CCW order, see if all circumcenters are empty.
\param i the target cell
\param neighbors the set of proposed neighbors
\param timing keep track of some timing info
*/
bool DelaunayLoc::testPointTriangulation(int i, vector<int> &neighbors, bool timing)
    {

    Dscalar2 v = pts[i];
    //for each circumcirlce, see if its empty
    int neigh1 = neighbors[neighbors.size()-1];
    vector<int> cns;
    Dscalar radius;
    Dscalar vx = 0.0; Dscalar vy = 0.0;
    bool repeat = false;
    Dscalar2 tocenter, disp;

    for (int nn = 0; nn < neighbors.size(); ++nn)
        {
        if (repeat) continue;
        int neigh2 = neighbors[nn];
        Dscalar2 pt1, pt2;
        Box->minDist(pts[neigh1],v,pt1);
        Box->minDist(pts[neigh2],v,pt2);

        Dscalar2 Q;
        Circumcircle(pt1,pt2,Q,radius);
        Dscalar rad2 = radius*radius;

        //what cell indices to check
        int cix = cList.positionToCellIndex(v.x+Q.x,v.y+Q.y);
        int wcheck = ceil(radius/cList.getBoxsize())+1;
        cList.getCellNeighbors(cix,wcheck,cns);
        ArrayHandle<unsigned int> h_cs(cList.cell_sizes,access_location::host,access_mode::read);
        ArrayHandle<int> h_idx(cList.idxs,access_location::host,access_mode::read);
        for (int cc = 0; cc < cns.size(); ++cc)
            {
            if (repeat) continue;
            int numberInCell = h_cs.data[cns[cc]];
            for (int pp = 0; pp < numberInCell;++pp)
                {
                if (repeat) continue;

                int idx = h_idx.data[cList.cell_list_indexer(pp,cns[cc])];
                Box->minDist(pts[idx],v,disp);
                //how far is the point from the circumcircle's center?
                Box->minDist(disp,Q,tocenter);
                if(tocenter.x*tocenter.x+tocenter.y*tocenter.y<rad2)
                    {
                    //double check that it isn't one of the points in the nlist or i
                    repeat = true;
                    for (int n2 = 0; n2 < neighbors.size();++n2)
                        if (neighbors[n2] == idx) repeat = false;
                    if (idx == i) repeat = false;
                    };
                };
            };
        neigh1 = neigh2;
        }; // end loop over neighbors for circumcircle


    return (!repeat);
    };

/*!
Test all circumcircles to see if they are empty, and flag particles for retriangulation if needed.
\param ccs a vector of length (3*numberOfCircumcircles)
\param points a vector of bools, where if true that point needs its neighbors re-triangulated
\param timing keep track of timing info if true
*/
void DelaunayLoc::testTriangulation(vector<int> &ccs, vector<bool> &points, bool timing)
    {
    Dscalar vx = 0.0; Dscalar vy = 0.0;
    int circumcircles = ccs.size()/3;

    for (int c = 0; c < circumcircles; ++c)
        {
        int ii = ccs[3*c];
        int neigh1 = ccs[3*c+1];
        int neigh2 = ccs[3*c+2];
        Dscalar2 v=pts[ii];
        Dscalar2 pt1,pt2;
        Box->minDist(pts[neigh1],v,pt1);
        Box->minDist(pts[neigh2],v,pt2);

        vector<int> cns;
        Dscalar radius;
        bool repeat = false;

        Dscalar2 tocenter,disp;
        Dscalar2 Q;
        Circumcircle(pt1,pt2,Q,radius);
        Dscalar rad2 = radius*radius;

        //what cell indices to check
        int cix = cList.positionToCellIndex(v.x+Q.x,v.y+Q.y);
        int wcheck = ceil(radius/cList.getBoxsize())+1;
        cList.getCellNeighbors(cix,wcheck,cns);

        ArrayHandle<unsigned int> h_cs(cList.cell_sizes,access_location::host,access_mode::read);
        ArrayHandle<int> h_idx(cList.idxs,access_location::host,access_mode::read);
        for (int cc = 0; cc < cns.size(); ++cc)
            {
            if (repeat) continue;
            int numberInCell = h_cs.data[cns[cc]];
            for (int pp = 0; pp < numberInCell;++pp)
                {
                if (repeat) continue;

                int idx = h_idx.data[cList.cell_list_indexer(pp,cns[cc])];
                Box->minDist(pts[idx],v,disp);
                //how far is the point from the circumcircle's center?
                Box->minDist(disp,Q,tocenter);
                if(tocenter.x*tocenter.x+tocenter.y*tocenter.y<rad2)
                    {
                    //double check that it isn't one of the points in the nlist or i
                    repeat = true;
                    if (idx == ii) repeat = false;
                    if (idx == neigh1) repeat = false;
                    if (idx == neigh2) repeat = false;
                    };
                };
            };

        if (repeat)
            {
            points[ii] = true;
            points[neigh1]=true;
            points[neigh2]=true;
            repeat = false;
            };

        }; // end loop over circumcircles
    };
/*!
A utility function for testing and debugging
\param maxprint the maximum number of points to print info for
*/
void DelaunayLoc::printTriangulation(int maxprint)
    {
    if (!triangulated)
        {
        cout << "No triangulation has been performed -- nothing to print" <<endl;
        return;
        };
    for (int ii = 0; ii < min(maxprint,nV); ++ii)
        {
        cout << pts[ii].x << "  " <<pts[ii].y << "  " <<ii << endl;
        };
    for (int tt = 0; tt < min(maxprint,DT.nTriangles); ++tt)
        cout <<"{"<<DT.triangles[tt].i << ",  " <<DT.triangles[tt].j << ",  " <<DT.triangles[tt].k <<"}" << endl;

    };

/*!
A utility function for testing and debugging
\param outfile the ofstream to write triangulation data to
*/
void DelaunayLoc::writeTriangulation(ofstream &outfile)
    {
    outfile << nV <<"\t"<<DT.nTriangles<<"\t"<<DT.nEdges<<endl;
    for (int ii = 0; ii < nV ; ++ii)
        outfile << pts[ii].x <<"\t" <<pts[ii].y <<endl;
    for (int ii = 0; ii < DT.nTriangles; ++ii)
        outfile << DT.triangles[ii].i <<"\t" <<DT.triangles[ii].j<<"\t"<<DT.triangles[ii].k<<endl;
    for (int ii = 0; ii < DT.nEdges; ++ii)
        outfile << DT.edges[ii].i <<"\t" <<DT.edges[ii].j<<endl;
    };


/*!
A utility function to test and time several routines in this class
\param numpts The number of points for which to perform tests
\param tmax the number of times each function will be called, for averaging timing info
\param err detect changes by moving points by some amount
\param verbose if true output even more timing info
*/
void DelaunayLoc::testDel(int numpts, int tmax,double err, bool verbose)
    {
    cout << "Timing DelaunayLoc routine..." << endl;
    nV = numpts;
    Dscalar boxa = sqrt(numpts)+1.0;
    gpubox Bx(boxa,boxa);
    setBox(Bx);
    vector<Dscalar> ps2(2*numpts);
    vector<Dscalar> ps3(2*numpts);
    Dscalar maxx = 0.0;
    int randmax = 1000000;
    for (int i=0;i<numpts;++i)
        {
        Dscalar x =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        Dscalar y =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        ps2[i*2]=x;
        ps2[i*2+1]=y;
        Dscalar x3 =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        Dscalar y3 =EPSILON+boxa/(Dscalar)randmax* (Dscalar)(rand()%randmax);
        ps3[i*2]=x3;
        ps3[i*2+1]=y3;
        };
    setPoints(ps2);
    initialize(boxa/sqrt(numpts));
    clock_t tstart,tstop;
    tstart = clock();

    vector<vector<int> > allneighs(numpts);
    vector<bool> reTriangulate(numpts,false);
    vector<int> circumcenters;
    circumcenters.reserve(2*numpts);


    geotiming=polytiming=ringcandtiming=reducedtiming=tritiming=tritesttiming=0.;
    Dscalar timing = 0.;
    for (int tt = 0; tt < tmax; ++tt)
        {
        setPoints(ps2);
        DelaunayCell cell;
        vector<int> neighs;
        circumcenters.clear();
        for (int nn = 0; nn < numpts; ++nn)
            {
            tstart = clock();
            triangulatePoint(nn,neighs,cell,true);
            allneighs[nn]=neighs;
            tstop=clock();
            timing += (tstop-tstart)/(Dscalar)CLOCKS_PER_SEC/(Dscalar)tmax;


            for (int jj = 0; jj < neighs.size();++jj)
                {
                int n1 = neighs[jj];
                int ne2 = jj + 1;
                if (jj == neighs.size()-1) ne2 = 0;
                int n2 = neighs[ne2];
                if (nn < n1 && nn < n2)
                    {
                    circumcenters.push_back(nn);
                    circumcenters.push_back(n1);
                    circumcenters.push_back(n2);
                    };

                };

            };
        for (int nn = 0; nn < ps2.size(); ++nn)
            {
            Dscalar diff = -err*0.5+err*(Dscalar)(rand()%randmax)/((Dscalar)randmax); 
            ps3[nn] = ps2[nn] + diff;
            };
        setPoints(ps3);
        vector<bool> reTriangulate(numpts,false);
        testTriangulation(circumcenters,reTriangulate,true);
        tstart = clock();
        for (int nn = 0; nn < numpts; ++nn)
            {
            if (reTriangulate[nn]==true) 
                {
                //do something if you want to test!
                };
            };
        tstop=clock();
        tritesttiming += (tstop-tstart)/(Dscalar)CLOCKS_PER_SEC;
        };

    totaltiming=timing;
    cout << "average time per complete triangulation = " << timing << endl;
    if (verbose)
        {
        cout << "          mean time for getPolygon = " << polytiming/(Dscalar)tmax << endl;
        cout << "          mean time for 1ringcandidates = " << (ringcandtiming-polytiming-reducedtiming)/(Dscalar)tmax << endl;
        cout << "          mean time for reduced ring= " << reducedtiming/(Dscalar)tmax << endl;
        cout << "          mean time for geometry   = " << geotiming/(Dscalar)tmax << endl;
        cout << "          mean time for triangulation   = " << tritiming/(Dscalar)tmax << endl;
        cout << "   ratio of total candidate time to triangulating time:  " <<ringcandtiming/tritiming << endl;
        cout << "average time to check triangulation   = " << tritesttiming/(Dscalar)tmax << endl;
        };

    };
