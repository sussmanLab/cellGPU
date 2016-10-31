#ifndef GPUCELLLIST
#define GPUCELLLIST

///


using namespace std;
using namespace voroguppy;

#include "gpubox.h"
#include "gpuarray.h"
#include "indexer.h"
//#include "gpucell.cuh"
//namespace voroguppy
//{

class cellListGPU
    {
    public:
        cellListGPU(){};
        cellListGPU(vector<float> &points);
        cellListGPU(dbl a, vector<float> &points, gpubox &bx);
        //~cellListGPU();

        void setParticles(const vector<float> &points);
        void setBox(gpubox &bx);
        void setNp(int nn){Np=nn;};

        //only call this if particles and box already set...doubles as a general initialization of data structures
        void setGridSize(dbl a);
        float getGridSize() {return boxsize;};
        int getNmax() {return Nmax;};
        int getXsize() {return xsize;};
        int getYsize() {return ysize;};
        float getBoxsize() {return boxsize;};

        //initialization and helper
        void resetCellSizes();

        const GPUArray<unsigned int>& getCellSizeArray() const
            {
            return cell_sizes;
            };
        const GPUArray<int>& getIdxArray() const
            {
            return idxs;
            };

        void compute(); // compute the cell list given current particle positions

        void computeGPU(); // compute the cell list given current particle positions

        void computeGPU(GPUArray<float2> &points); // compute the cell list of the gpuarry passed to it


        void repP(int i)
            {
            if(true){
            ArrayHandle<float2> hh(particles,access_location::host,access_mode::read);
            cout <<hh.data[i].x << "  " << hh.data[i].y << endl;
            };
            };
        
        Index2D cell_indexer; //indexes cells from (i,j) pairs
        Index2D cell_list_indexer; //indexes elements in the cell list

        GPUArray<float2> particles;
        GPUArray<unsigned int> cell_sizes; //number of elements in each cell
        GPUArray<int> idxs; //cell list with index

    private:
        GPUArray<int> assist; //first index is Nmax, second is whether to recompute
        int Np; //number of particles to put in cells
        dbl boxsize; //linear size of each grid cell
        int xsize;
        int ysize;
        int totalCells;
        int Nmax; //space reserved for particles in each cell
        gpubox Box;

    };


//};

#endif
