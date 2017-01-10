#ifndef CELLLIST
#define CELLLIST

//cell_list class...based on (inefficient) vector implementation. Will switch to linked lists later.
//only works for square boxes


using namespace std;
#include "std_include.h"
#include "structures.h"
#include "gpubox.h"

/*!
The grid class implements a simple cell/bucket structure. It is currently used by the DelaunayLoc and DelaunayNP classes, but should be deprecated in favor of a general cellListGPU class that functions both for CPU and GPU calculations. Since the lifetime should be short, I will not document this class. Every time I read these sentences I will feel guilty and get a nudge to actually refactor the code.
As an important note, this and cellListGPU are currently restricted to square boxes.
*/
//!A simple cell/bucket structure
class grid
    {
    private:
        std::vector<Dscalar2> points;
        Dscalar cellsize;
        int N;
        int cellnumx, cellnumy,totalCells;
        gpubox Box;
    public:
        //The fundamental structure
        std::vector< std::vector<int> > cells;

        inline grid(){};
        inline grid(std::vector<Dscalar2> &pts, gpubox &bx, Dscalar cs)
            {points=pts;cellsize=cs;setBox(bx);initialize();};

        inline void setPoints(std::vector<Dscalar2> &pts){points = pts;};
        inline void setCellSize(Dscalar cs){cellsize = cs;};
        inline void setBox(gpubox &bx);

        inline void initialize();
        inline void construct();

        inline int getN(){return N;};
        inline int getNx(){return cellnumx;};
        inline Dscalar getCellSize(){return cellsize;};
        inline int posToCellIdx(Dscalar x, Dscalar y);
        //all cells within width of the cell index
        inline void cellNeighbors(int cidx, int width, std::vector<int> &cellneighs);
        inline void cellNeighborsShort(int cidx, int width, std::vector<int> &cellneighs);
        //all cells in square shells aroudn the target cell index
        inline void cellShell(int cidx, int width, std::vector<int> &cellneighs);
        //all particle indices that are in the cell
        inline void getParticles(int cidx, std::vector<int> &plist);
    };

void grid::setBox(gpubox &bx)
    {
    Dscalar b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    Box.setGeneral(b11,b12,b21,b22);
    };

void grid::initialize()
    {
    N = points.size();
    Dscalar bx,bxx,by,byy;
    Box.getBoxDims(bx,bxx,byy,by);
    cellnumx = (int)floor(bx/cellsize);
    if(cellnumx%2==1) cellnumx +=1;
    cellnumy= (int) floor(by/cellsize);
    if(cellnumy%2==1) cellnumy +=1;
    totalCells = cellnumx*cellnumy;
    cellsize = bx/cellnumx;
    cells.clear();
    cells.resize(totalCells);
    };

void grid::construct()
    {
    cells.clear();
    cells.resize(totalCells);
    for (int nn = 0; nn < N; ++nn)
        {
        int idx = posToCellIdx(points[nn].x,points[nn].y);
//        if (idx >= cells.size() || idx < 0)
//            cout << "bad idx for particle" << nn << " : " << idx << "  " << points[nn].x << "  " << points[nn].y << endl; cout.flush();
        cells[idx].push_back(nn);
        };
    };

int grid::posToCellIdx(Dscalar x, Dscalar y)
    {
    int cell_idx = 0;
    int binx = max(0,min(cellnumx-1,(int)floor(x/cellsize)));
    int biny = max(0,min(cellnumx-1,(int)floor(y/cellsize)));
    cell_idx += binx;
    cell_idx += cellnumx*(biny);
    return cell_idx;
    };

void grid::cellNeighbors(int cidx, int width, std::vector<int> &cellneighs)
    {
    int w = min(width,cellnumx/2);
    int cellix = cidx%cellnumx;
    int celliy = (cidx - cellix)/cellnumx;
    cellneighs.clear();
    cellneighs.reserve(w*w);
    for (int ii = -w; ii <=w; ++ii)
        for (int jj = -w; jj <=w; ++jj)
            {
            int cx = (cellix+jj)%cellnumx;
            if (cx <0) cx+=cellnumx;
            int cy = (celliy+ii)%cellnumx;
            if (cy <0) cy+=cellnumx;
            cellneighs.push_back(cx+cellnumx*cy);
            };
    };

void grid::cellNeighborsShort(int cidx, int width, std::vector<int> &cellneighs)
    {
    int w = min(width,cellnumx/2);
    int cellix = cidx%cellnumx;
    int celliy = (cidx - cellix)/cellnumx;
    cellneighs.clear();
    cellneighs.reserve(w*w);
    for (int ii = -w; ii <=w; ++ii)
        for (int jj = -w; jj <=w; ++jj)
            {
            int cx = (cellix+jj)%cellnumx;
            if (cx <0) cx+=cellnumx;
            int cy = (celliy+ii)%cellnumx;
            if (cy <0) cy+=cellnumx;
            int idx = cx+cellnumx*cy;
            if (cells[idx].size()>0)
                cellneighs.push_back(idx);
            };
    };

void grid::cellShell(int cidx, int width, std::vector<int> &cellneighs)
    {
    int w = min(width,cellnumx);
    int cellix = cidx%cellnumx;
    int celliy = (cidx - cellix)/cellnumx;
    cellneighs.clear();
    for (int ii = -w; ii <=w; ++ii)
        for (int jj = -w; jj <=w; ++jj)
            {
            if(ii ==-w ||ii == w ||jj ==-w ||jj==w)
                {
                int cx = (cellix+jj)%cellnumx;
                if (cx <0) cx+=cellnumx;
                int cy = (celliy+ii)%cellnumx;
                if (cy <0) cy+=cellnumx;
                cellneighs.push_back(cx+cellnumx*cy);
                };
            };
    };

void grid::getParticles(int cidx, std::vector<int> &plist)
    {
    plist = cells[cidx];
    };

#endif
