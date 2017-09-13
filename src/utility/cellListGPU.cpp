#define ENABLE_CUDA

#include "std_include.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"
#include "cuda_runtime.h"
#include "cellListGPU.cuh"
#include "cellListGPU.h"
/*! \file cellListGPU.cpp */

/*!
\param a the approximate side length of the cells
\param points the positions of points to populate the cell list with
\param bx the period box for the system
 */
cellListGPU::cellListGPU(Dscalar a, vector<Dscalar> &points,gpubox &bx)
    {
    Nmax = 0;
    setParticles(points);
    Box = make_shared<gpubox>();
    setGridSize(a);
    }

/*!
\param points the positions of points to populate the cell list with
 */
cellListGPU::cellListGPU(vector<Dscalar> &points)
    {
    Nmax = 0;
    Box = make_shared<gpubox>();
    setParticles(points);
    }

/*!
\param nn the number of particles to sort
 */
void cellListGPU::setNp(int nn)
    {
    Np = nn;
    };

/*!
\param points set the list of points cellListGPU knows about to this vector
 */
void cellListGPU::setParticles(const vector<Dscalar> &points)
    {
    int newsize = points.size()/2;
    particles.resize(newsize);
    Np=newsize;
    if(true)
        {
        ArrayHandle<Dscalar2> h_handle(particles,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < points.size()/2; ++ii)
            {
            h_handle.data[ii].x = points[2*ii];
            h_handle.data[ii].y = points[2*ii+1];
            };
        };
    };

/*!
\param points set the list of points cellListGPU knows about to this vector of Dscalar2's
 */
void cellListGPU::setParticles(const vector<Dscalar2> &points)
    {
    int newsize = points.size();
    particles.resize(newsize);
    Np=newsize;
    if(true)
        {
        ArrayHandle<Dscalar2> h_handle(particles,access_location::host,access_mode::overwrite);
        for (int ii = 0; ii < points.size(); ++ii)
            {
            h_handle.data[ii] = points[ii];
            };
        };
    };

/*!
\param bx the box defining the periodic unit cell
 */
void cellListGPU::setBox(gpubox &bx)
    {
    Dscalar b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    if (bx.isBoxSquare())
        Box->setSquare(b11,b22);
    else
        Box->setGeneral(b11,b12,b21,b22);
    };

/*!
\param a the approximate side length of all of the cells.
This routine currently picks an even integer of cells, close to the desired size, that fit in the box.
 */
void cellListGPU::setGridSize(Dscalar a)
    {
    Dscalar b11,b12,b21,b22;
    Box->getBoxDims(b11,b12,b21,b22);
    xsize = (int)floor(b11/a);
    if(xsize%2==1) xsize +=1;
    ysize = (int)floor(b22/a);
    if(ysize%2==1) ysize +=1;

    boxsize = b11/xsize;

    totalCells = xsize*ysize;
    cell_sizes.resize(totalCells); //number of elements in each cell...initialize to zero

    cell_indexer = Index2D(xsize,ysize);

    //estimate Nmax
    if(ceil(Np/totalCells)+1 > Nmax)
        Nmax = ceil(Np/totalCells)+1;
    resetCellSizesCPU();
    };

/*!
Sets all cell sizes to zero, all cell indices to zero, and resets the "assist" utility structure,
all on the CPU (so that no expensive copies are needed)
 */
void cellListGPU::resetCellSizesCPU()
    {
    //set all cell sizes to zero
    totalCells=xsize*ysize;
    if(cell_sizes.getNumElements() != totalCells)
        cell_sizes.resize(totalCells);

    ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::overwrite);
    for (int i = 0; i <totalCells; ++i)
        h_cell_sizes.data[i]=0;

    //set all cell indexes to zero
    cell_list_indexer = Index2D(Nmax,totalCells);
    if(idxs.getNumElements() != cell_list_indexer.getNumElements())
        idxs.resize(cell_list_indexer.getNumElements());

    ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::overwrite);
    for (int i = 0; i < cell_list_indexer.getNumElements(); ++i)
        h_idx.data[i]=0;

    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
    };


/*!
Sets all cell sizes to zero, all cell indices to zero, and resets the "assist" utility structure,
all on the GPU so that arrays don't need to be copied back to the host
*/
void cellListGPU::resetCellSizes()
    {
    //set all cell sizes to zero
    totalCells=xsize*ysize;
    if(cell_sizes.getNumElements() != totalCells)
        cell_sizes.resize(totalCells);

    ArrayHandle<unsigned int> d_cell_sizes(cell_sizes,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_cell_sizes.data,totalCells);

    //set all cell indexes to zero
    cell_list_indexer = Index2D(Nmax,totalCells);
    if(idxs.getNumElements() != cell_list_indexer.getNumElements())
        idxs.resize(cell_list_indexer.getNumElements());

    ArrayHandle<int> d_idx(idxs,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_idx.data,(int) cell_list_indexer.getNumElements());


    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
    };

/*!
\param x the x coordinate of the position
\param y the y coordinate of the position
returns the cell index that (x,y) would be contained in for the current cell list
 */
int cellListGPU::positionToCellIndex(Dscalar x, Dscalar y)
    {
    int cell_idx = 0;
    int binx = max(0,min(xsize-1,(int)floor(x/boxsize)));
    int biny = max(0,min(ysize-1,(int)floor(y/boxsize)));
    return cell_indexer(binx,biny);
    };

/*!
\param cellIndex the base cell index to find the neighbors of
\param width the distance (in cells) to search
\param cellNeighbors a vector of all cell indices that are neighbors of cellIndex
 */
void cellListGPU::getCellNeighbors(int cellIndex, int width, std::vector<int> &cellNeighbors)
    {
    int w = min(width,xsize/2);
    int cellix = cellIndex%xsize;
    int celliy = (cellIndex - cellix)/ysize;
    cellNeighbors.clear();
    cellNeighbors.reserve(w*w);
    for (int ii = -w; ii <=w; ++ii)
        for (int jj = -w; jj <=w; ++jj)
            {
            int cx = (cellix+jj)%xsize;
            if (cx <0) cx+=xsize;
            int cy = (celliy+ii)%ysize;
            if (cy <0) cy+=ysize;
            cellNeighbors.push_back(cell_indexer(cx,cy));
            };
    };

/*!
\param cellIndex the base cell index to find the neighbors of
\param width the distance (in cells) to search
\param cellNeighbors a vector of all cell indices that are neighbors of cellIndex
This method returns a square outline of neighbors (the neighbor shell) rather than all neighbors
within a set distance
 */
void cellListGPU::getCellShellNeighbors(int cellIndex, int width, std::vector<int> &cellNeighbors)
    {
    int w = min(width,xsize);
    int cellix = cellIndex%xsize;
    int celliy = (cellIndex - cellix)/xsize;
    cellNeighbors.clear();
    for (int ii = -w; ii <=w; ++ii)
        for (int jj = -w; jj <=w; ++jj)
            if(ii ==-w ||ii == w ||jj ==-w ||jj==w)
                {
                int cx = (cellix+jj)%xsize;
                if (cx <0) cx+=xsize;
                int cy = (celliy+ii)%ysize;
                if (cy <0) cy+=ysize;
                cellNeighbors.push_back(cell_indexer(cx,cy));
                };
    };


/*!
This puts the points the cellList currently knows about into cells
 */
void cellListGPU::compute()
    {
    //will loop through particles and put them in cells...
    //if there are more than Nmax particles in any cell, will need to recompute.
    bool recompute = true;
    ArrayHandle<Dscalar2> h_pt(particles,access_location::host,access_mode::read);
    int ibin, jbin;
    int nmax = Nmax;
    int computations = 0;
    while (recompute)
        {
        //reset particles per cell, reset cell_list_indexer, resize idxs
        resetCellSizesCPU();
        ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::readwrite);
        ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::readwrite);
        recompute=false;

        for (int nn = 0; nn < Np; ++nn)
            {
            if (recompute) continue;
            ibin = floor(h_pt.data[nn].x/boxsize);
            jbin = floor(h_pt.data[nn].y/boxsize);
            int bin = cell_indexer(ibin,jbin);
            int offset = h_cell_sizes.data[bin];
            if (offset < Nmax)
                {
                int clpos = cell_list_indexer(offset,bin);
                h_idx.data[cell_list_indexer(offset,bin)]=nn;
                }
            else
                {
                nmax = max(Nmax,offset+1);
                Nmax=nmax;
                recompute=true;
                };
            h_cell_sizes.data[bin]++;
            };
        computations++;
        };
    cell_list_indexer = Index2D(Nmax,totalCells);
    };


/*!
\param points the set of points to assign to cells
 */
void cellListGPU::compute(GPUArray<Dscalar2> &points)
    {
    //will loop through particles and put them in cells...
    //if there are more than Nmax particles in any cell, will need to recompute.
    bool recompute = true;
    ArrayHandle<Dscalar2> h_pt(points,access_location::host,access_mode::read);
    int ibin, jbin;
    int nmax = Nmax;
    int computations = 0;
    while (recompute)
        {
        //reset particles per cell, reset cell_list_indexer, resize idxs
        resetCellSizesCPU();
        ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::readwrite);
        ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::readwrite);
        recompute=false;

        for (int nn = 0; nn < Np; ++nn)
            {
            if (recompute) continue;
            ibin = floor(h_pt.data[nn].x/boxsize);
            jbin = floor(h_pt.data[nn].y/boxsize);

            int bin = cell_indexer(ibin,jbin);
            int offset = h_cell_sizes.data[bin];
            if (offset < Nmax)
                {
                int clpos = cell_list_indexer(offset,bin);
                h_idx.data[cell_list_indexer(offset,bin)]=nn;
                }
            else
                {
                nmax = max(Nmax,offset+1);
                Nmax=nmax;
                recompute=true;
                };
            h_cell_sizes.data[bin]++;
            };
        computations++;
        };
    cell_list_indexer = Index2D(Nmax,totalCells);
    };


/*!
Assign known points to cells on the GPU
 */
void cellListGPU::computeGPU()
    {
    bool recompute = true;
    //resetCellSizes();

    while (recompute)
        {
        //cout << "computing cell list on the gpu with Nmax = " << Nmax << endl;
        resetCellSizes();

        //scope for arrayhandles
        if (true)
            {
            //get particle data
            ArrayHandle<Dscalar2> d_pt(particles,access_location::device,access_mode::read);

            //get cell list arrays...readwrite so things are properly zeroed out
            ArrayHandle<unsigned int> d_cell_sizes(cell_sizes,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_idx(idxs,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);

            //call the gpu function
            gpu_compute_cell_list(d_pt.data,        //particle positions...broken
                          d_cell_sizes.data,//particles per cell
                          d_idx.data,       //cell list
                          Np,               //number of particles
                          Nmax,             //maximum particles per cell
                          xsize,            //number of cells in x direction
                          ysize,            // ""     ""      "" y directions
                          boxsize,          //size of each grid cell
                          (*Box),
                          cell_indexer,
                          cell_list_indexer,
                          d_assist.data
                          );               //the box
            }
        //get cell list arrays
        recompute = false;
        //bool loopcheck=false;
        if (true)
            {

            ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::read);
            ArrayHandle<int> h_idx(idxs,access_location::host,access_mode::read);
            for (int cc = 0; cc < totalCells; ++cc)
                {
                int cs = h_cell_sizes.data[cc] ;
                for (int bb = 0; bb < cs; ++bb)
                    {
                    int wp = cell_list_indexer(bb,cc);
                    };
                if(cs > Nmax)
                    {
                    Nmax =cs ;
                    recompute = true;
                    };

                };

            };
        };
    cell_list_indexer = Index2D(Nmax,totalCells);

    };

/*!
\param points the set of points to assign to cells...on the GPU
 */
void cellListGPU::computeGPU(GPUArray<Dscalar2> &points)
    {
    bool recompute = true;
    resetCellSizes();

    while (recompute)
        {
        //cout << "computing cell list on the gpu with Nmax = " << Nmax << endl;
        resetCellSizes();
        //scope for arrayhandles
        if (true)
            {
            //get particle data
            ArrayHandle<Dscalar2> d_pt(points,access_location::device,access_mode::read);

            //get cell list arrays...readwrite so things are properly zeroed out
            ArrayHandle<unsigned int> d_cell_sizes(cell_sizes,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_idx(idxs,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);

            //call the gpu function
            gpu_compute_cell_list(d_pt.data,        //particle positions...broken
                          d_cell_sizes.data,//particles per cell
                          d_idx.data,       //cell list
                          Np,               //number of particles
                          Nmax,             //maximum particles per cell
                          xsize,            //number of cells in x direction
                          ysize,            // ""     ""      "" y directions
                          boxsize,          //size of each grid cell
                          (*Box),
                          cell_indexer,
                          cell_list_indexer,
                          d_assist.data
                          );               //the box
            }
        //get cell list arrays
        recompute = false;
        if (true)
            {
            ArrayHandle<unsigned int> h_cell_sizes(cell_sizes,access_location::host,access_mode::read);
            for (int cc = 0; cc < totalCells; ++cc)
                {
                int cs = h_cell_sizes.data[cc] ;
                if(cs > Nmax)
                    {
                    Nmax =cs ;
                    if (Nmax%2 == 0 ) Nmax +=2;
                    if (Nmax%2 == 1 ) Nmax +=1;
                    recompute = true;
                    };

                };

            };
        };
    cell_list_indexer = Index2D(Nmax,totalCells);
    };
