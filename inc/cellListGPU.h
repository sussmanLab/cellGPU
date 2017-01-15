#ifndef GPUCELLLIST
#define GPUCELLLIST

#include "std_include.h"
#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"


/*!
 * A class that can sort points into a grid of buckets. This enables local searches for particle neighbors, etc.
 * Note that at the moment this class (and some of the classes above it) can only handle square boxes. This is
 * not a fundamental limitation, though.
 \todo expand to work correctly with non-square boxes
 */
//! Construct simple cell/bucket structures on the GPU, using kernels in \ref cellListGPUKernels
class cellListGPU
    {
    public:
        //!Blank constructor
        cellListGPU(){};
        //!construct with a given set of points
        cellListGPU(vector<Dscalar> &points);
        //! constructor with points, a box, and a size for the underlying grid
        cellListGPU(Dscalar a, vector<Dscalar> &points, gpubox &bx);

        //!Set the object's points to a given vector
        void setParticles(const vector<Dscalar> &points);
        //!Set the objects box object
        void setBox(gpubox &bx);
        //!Set the number of particles to put in the buckets
        void setNp(int nn){Np=nn;};

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

        //!Initialization and helper
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
        void compute(); // compute the cell list given current particle positions

        //!Compute the cell list for the class' GPUArray of particles on the GPU
        void computeGPU();

        void computeGPU(GPUArray<Dscalar2> &points); //!< compute the cell list of the gpuarry passed to it. GPU function
        void compute(GPUArray<Dscalar2> &points); //!< compute the cell list of the gpuarry passed to it. GPU function

        //!A debugging function to report where a point is
        void repP(int i)
            {
            if(true)
                {
                ArrayHandle<Dscalar2> hh(particles,access_location::host,access_mode::read);
                cout <<hh.data[i].x << "  " << hh.data[i].y << endl;
                };
            };

        Index2D cell_indexer; //!< Indexes the cells in the grid themselves (so the bin corresponding to the (j,i) position of the grid is bin=cell_indexer(i,j))
        Index2D cell_list_indexer; //!<Indexes elements in the cell list

        GPUArray<Dscalar2> particles; //!<The particles that some methods act on
        GPUArray<unsigned int> cell_sizes; //!< An array containing the number of elements in each cell
        GPUArray<int> idxs; //!<An array containing the indices of particles in various cells. So, idx[cell_list_indexer(nn,bin)] gives the index of the nth particle in the bin "bin" of the cell list

    private:
        GPUArray<int> assist; //!<first index is Nmax, second is whether to recompute
        int Np; //!<THe number of particles to put in cells
        Dscalar boxsize; //!< The linear size of each grid cell
        int xsize; //!<The number of bins in the x-direction
        int ysize; //!the number of bins in the y-direction
        int totalCells; //!xsize*ysize
        int Nmax; //!< the maximum number of particles found in any bin
        gpubox Box; //!<The Box used to compute periodic distances

    };


#endif
