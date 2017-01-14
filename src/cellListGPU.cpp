#define ENABLE_CUDA

#include "std_include.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"
#include "cuda_runtime.h"
#include "cellListGPU.cuh"
#include "cellListGPU.h"


cellListGPU::cellListGPU(Dscalar a, vector<Dscalar> &points,gpubox &bx)
    {
    setParticles(points);
    setBox(bx);
    setGridSize(a);
    }
cellListGPU::cellListGPU(vector<Dscalar> &points)
    {
    setParticles(points);
    }

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

void cellListGPU::setBox(gpubox &bx)
    {
    Dscalar b11,b12,b21,b22;
    bx.getBoxDims(b11,b12,b21,b22);
    if (bx.isBoxSquare())
        Box.setSquare(b11,b22);
    else
        Box.setGeneral(b11,b12,b21,b22);
    };

void cellListGPU::setGridSize(Dscalar a)
    {
    Dscalar b11,b12,b21,b22;
    Box.getBoxDims(b11,b12,b21,b22);
    xsize = (int)floor(b11/a);
    ysize = (int)floor(b22/a);

    boxsize = b11/xsize;

    xsize = (int)ceil(b11/boxsize);
    ysize = (int)ceil(b22/boxsize);

    totalCells = xsize*ysize;
    cell_sizes.resize(totalCells); //number of elements in each cell...initialize to zero

    cell_indexer = Index2D(xsize,ysize);

    //estimate Nmax
    Nmax = ceil(Np/totalCells)+1;
    resetCellSizesCPU();
    };

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
        resetCellSizes();
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
                          Box,
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
                          Box,
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

