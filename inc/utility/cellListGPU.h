#ifndef GPUCELLLIST
#define GPUCELLLIST

#include "std_include.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"


/*! \file cellListGPU.h */
//! Construct simple cell/bucket structures on the GPU, using kernels in \ref cellListGPUKernels
/*!
 * A class that can sort points into a grid of buckets. This enables local searches for particle neighbors, etc.
 * Note that at the moment this class (and some of the classes above it) can only handle square boxes. This is
 * not a fundamental limitation, though.
 */
class cellListGPU
    {
    public:
        //!Blank constructor
        cellListGPU(){Nmax=0;Box = make_shared<gpubox>();};
        //!construct with a given set of points
        cellListGPU(vector<Dscalar> &points);
        //! constructor with points, a box, and a size for the underlying grid
        cellListGPU(Dscalar a, vector<Dscalar> &points, gpubox &bx);

        //!Set the object's points to a given vector
        void setParticles(const vector<Dscalar> &points);
        //!Set the object's points to a given vector of Dscalar2s
        void setParticles(const vector<Dscalar2> &points);
        //!Set the objects box object
        void setBox(gpubox &bx);
        //!Set the BoxPtr to point to an existing one
        void setBox(BoxPtr bx){Box=bx;};
        //!Set the number of particles to put in the buckets
        void setNp(int nn);

        //!call setGridSize if the particles and box already set, as this doubles as a general initialization of data structures
        void setGridSize(Dscalar a);
        //!Get am upper bound on the maximum number of particles in a given bucket
        int getNmax() {return Nmax;};
        //!The number of cells in the x-direction
        int getXsize() {return xsize;};
        //!The number of cells in the y-direction
        int getYsize() {return ysize;};
        //!Returns the length of the square that forms the base grid size
        Dscalar getBoxsize() {return boxsize;};

        //!If the grid is already initialized, given a spatial position return the cell index
        int positionToCellIndex(Dscalar x,Dscalar y);
        //! given a target cell and a width, get all cell indices that sit in the surrounding square
        void getCellNeighbors(int cellIndex, int width, vector<int> &cellNeighbors);
        //! given a target cell and a width, get all cell indices that sit on the requested shell
        void getCellShellNeighbors(int cellIndex, int width, vector<int> &cellNeighbors);

        //!Initialization and helper without using the GPU
        void resetCellSizesCPU();
        //!Initialization and helper
        void resetCellSizes();
        //!Return the array of particles per cell
        const GPUArray<unsigned int>& getCellSizeArray() const
            {
            return cell_sizes;
            };
        //!Return the array of cell indices in the different cells
        const GPUArray<int>& getIdxArray() const
            {
            return idxs;
            };

        //!Compute the cell list on the CPU, given the current particle positions in the GPUArray of particles
        void compute();

        //!Compute the cell list for the class' GPUArray of particles on the GPU
        void computeGPU();

        //! compute the cell list of the gpuarry passed to it. GPU function
        void computeGPU(GPUArray<Dscalar2> &points);
        //! compute the cell list of the gpuarry passed to it. GPU function
        void compute(GPUArray<Dscalar2> &points);

        //!A debugging function to report where a point is
        void repP(int i)
            {
            if(true)
                {
                ArrayHandle<Dscalar2> hh(particles,access_location::host,access_mode::read);
                cout <<hh.data[i].x << "  " << hh.data[i].y << endl;
                };
            };

        //! Indexes the cells in the grid themselves (so the bin corresponding to the (j,i) position of the grid is bin=cell_indexer(i,j))
        Index2D cell_indexer;
        //!Indexes elements in the cell list
        Index2D cell_list_indexer;

        //!The particles that some methods act on
        GPUArray<Dscalar2> particles;
        //! An array containing the number of elements in each cell
        GPUArray<unsigned int> cell_sizes;
        //!An array containing the indices of particles in various cells. So, idx[cell_list_indexer(nn,bin)] gives the index of the nth particle in the bin "bin" of the cell list
        GPUArray<int> idxs;

    protected:
        //!first index is Nmax, second is whether to recompute
        GPUArray<int> assist;
        //!The number of particles to put in cells
        int Np;
        //! The linear size of each grid cell
        Dscalar boxsize;
        //!The number of bins in the x-direction
        int xsize;
        //!the number of bins in the y-direction
        int ysize;
        //!xsize*ysize
        int totalCells;
        //! the maximum number of particles found in any bin
        int Nmax;
        //!The Box used to compute periodic distances
        BoxPtr Box;
    };

#endif
