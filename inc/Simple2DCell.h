#ifndef SIMPLE2DCELL_H
#define SIMPLE2DCELL_H

#include "std_include.h"
#include "gpuarray.h"
#include "gpubox.h"
#include "curand.h"
#include "curand_kernel.h"

/*!
A class defining some of the fundamental attributes and operations common to 2D off-lattice models
of cells.
This class will help refactor the AVM and SPV branches into a more coherent set, rather than
kludging a solution with AVM based off of DelaunayMD but not using most of its concepts
*/

class Simple2DCell
    {
    //public functions first
    public:
        //!Currently a vacant constructor
        Simple2DCell();

        //!Set the simulation time stepsize
        void setDeltaT(Dscalar dt){deltaT = dt;};

        //!Enforce CPU-only operation.
        void setCPU(){GPUcompute = false;};

        //!Set uniform cell area and perimeter preferences
        void setCellPreferencesUniform(Dscalar A0, Dscalar P0);

        //!Set random cell positions, and set the periodic box to a square with average cell area=1
        void setCellPositionsRandomly();

        //!initialize the cuda RNG
        void initializeCurandStates(int gs, int i);

    //protected functions
    protected:

    //public member variables
    public:
        //!Number of cells in the simulation
        int Ncells;
        //!Number of vertices (i.e, degrees of freedom)
        int Nvertices;

        //!the box defining the periodic domain
        gpubox Box;

        //!A flag that, when true, has performTimestep call the GPU routines
        bool GPUcompute;

        //! Count the number of times "performTimeStep" has been called
        int Timestep;
        //!The time stepsize of the simulation
        Dscalar deltaT;
        //! Cell positions... not used for computation, but can track, e.g., MSD of cell centers
        GPUArray<Dscalar2> cellPositions;
        //! Position of the vertices
        GPUArray<Dscalar2> vertexPositions;

    //protected member variables
    protected:
        //!A flag to determine whether the CUDA RNGs should be initialized or not (so that the program will run on systems with no GPU by setting this to false
        bool initializeGPURNG;
        //!An array random-number-generators for use on the GPU branch of the code
        GPUArray<curandState> cellRNGs;
        //! A flag that determines whether the GPU RNG is the same every time.
        bool Reproducible;

        //!The current area and perimeter of each cell
        GPUArray<Dscalar2> AreaPeri;//(current A,P) for each cell
        //!The area and perimeter preferences of each cell
        GPUArray<Dscalar2> AreaPeriPreferences;//(A0,P0) for each cell

    //reporting functions
    public:
        //!report the current total area, and optionally the area and perimeter for each cell
        void reportAP(bool verbose = false)
                {
                ArrayHandle<Dscalar2> ap(AreaPeri,access_location::host,access_mode::read);
                Dscalar vtot= 0.0;
                for (int i = 0; i < Ncells; ++i)
                    {
                    if(verbose)
                        printf("%i: (%f,%f)\n",i,ap.data[i].x,ap.data[i].y);
                    vtot+=ap.data[i].x;
                    };
                printf("total area = %f\n",vtot);
                };
    };

#endif
