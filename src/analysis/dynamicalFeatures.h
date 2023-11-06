#ifndef dynamicalFeatures_H
#define dynamicalFeatures_H

#include "std_include.h"
#include "functions.h"
#include "periodicBoundaries.h"
#include "indexer.h"

/*! \file dynamicalFeatures.h */

//! A class that calculates various dynamical features for 2D systems
class dynamicalFeatures
    {
    public:
        //!The constructor takes in a defining set of boundary conditions
        dynamicalFeatures(GPUArray<double2> &initialPos, PeriodicBoxPtr _bx, double fractionAnalyzed = 1.0);

        //!set the list of neighbors forming initial cages of the particles (to be used for cage-relative calculations)
        void setCageNeighbors(GPUArray<int> &neighbors, GPUArray<int> &neighborNum, Index2D n_idx);

        //!Compute the mean squared displacement of the passed vector from the initial positions
        double computeMSD(GPUArray<double2> &currentPos);

        //!compute the overlap function
        double computeOverlapFunction(GPUArray<double2> &currentPos, double cutoff = 0.5);
        //!compute cage relative SISF with 2D angular averaging
        double computeSISF(GPUArray<double2> &currentPos, double k = 6.28319);

        //!compute cage relative MSD
        double computeCageRelativeMSD(GPUArray<double2> &currentPos);
        //!compute cage relative SISF with 2D angular averaging
        double computeCageRelativeSISF(GPUArray<double2> &currentPos, double k = 6.28319);

        //!compute chi_4 and F_s (result.x = Fs, result.y=chi_4)
        double2 computeFsChi4(GPUArray<double2> &currentPos, double k = 6.28319);
        //!compute cage-relative verions of above function
        double2 computeCageRelativeFsChi4(GPUArray<double2> &currentPos, double k = 6.28319);

        //!compute *un-normalized* flenner-Szamel psi_6 bond correlation decay (i.e., without the average |\psi_6|^2) that would make the function 1 at t=0. return.x is real, return.y is imaginary part
        double2 computeOrientationalCorrelationFunction(GPUArray<double2> &currentPos,GPUArray<int> &currentNeighbors, GPUArray<int> &currentNeighborNum, Index2D n_idx, int n=6);
        

    protected:
        //!a helper function that computes vectors of current displacements and cage relative displacements
        void computeCageRelativeDisplacements(GPUArray<double2> &currentPos);
        //!a helper function that computes vectors of current displacements
        void computeDisplacements(GPUArray<double2> &currentPos);
        //!helper function that computes the angular average of <F_s^2(q,t)>
        double chi4Helper(vector<double2> &displacements, double k);
        //!helper function that computes the angular average self-intermediate scattering function associated with a vector of displacements
        double angularAverageSISF(vector<double2> &displacements, double k);
        //!helper function that computes the mean dot product of a vector of double2's
        double MSDhelper(vector<double2> &displacements);
        //!the box defining the periodic domain
        PeriodicBoxPtr Box;
        //!the initial positions
        vector<double2> iPos;
        //! a vector of displacements relative to the initializing positions 
        vector<double2> currentDisplacements;
        //!the vector of current cage releative displacements
        vector<double2> cageRelativeDisplacements;
        //!the number of double2's
        int N;
        vector<vector<int>> cageNeighbors;
        Index2D nIdx;

        bool initialBondOrderComputed = false;
        vector<double2> initialConjugateBondOrder;
    };
#endif
