#ifndef autocorrelator_H
#define autocorrelator_H

#include "std_include.h"

/*! \file autocorrelator.h */

//! A class that allows efficient on-the-fly computation of auto-correlation functions
/*!
This class provides a multiple-time autocorrelation scheme. It is a generalization of the algorithm
presented in Frenkel and Smit (with independently controllable lag times), as explained in:
"Efficient on the fly calculation of time correlation functions in computer simulations"
J. Ramirez, S. K. Sukumaran, B. Vorselaars, and A. E. Likhtman
The Journal of Chemical Physics 133, 154103 (2010).

A primary modification of the source code linked to from that paper is that this class does not need
to set a maximum number of correlator levels; the correlator structure simply grows whenever it needs to
*/
class autocorrelator
    {
    public:
        //! The basic constructor
        autocorrelator(int pp=16, int mm=2, double deltaT = 1.0);
        //! Add a new data point to the correlator
        void add(double w, int k=0);
        //! evaluate the current state of the correlator... only set normalize to "true" if you need to correct for a non-zero bias (a la MSDs...details in the paper above)
        void evaluate(bool normalize = false);

        //!After evaluate is called, correlator is filled with the current autocorrelation.
        vector<double2> correlator;

        //! Set the time spacing
        void setDeltaT(double deltaT){dt=deltaT;};
        //! Initialize all data structures to zero
        void initialize();
        //! grow data structures if a new level of the correlation function needs to be added
        void growCorrelationLevel();

    protected:
        //!The time spacing
        double dt;
        //!points per correlator
        int p;
        //!number of points over which to average
        int m;
        //!Current number of correlator levels
        int nCorrelators;
        //!Accumulated sum of values added to the correlator
        double accumulatedValue;

        //!Minimum distance betwee points for correlators other than the zeroth level
        int minimumDistance;
        //!A vector saying where the currently added value should be inserted
        vector<int> insertIndex;
        //!A vector that helps control the accumulation in each correlator
        vector<int> nAccumulator;
        //!The actual values accumulating in each correlator
        vector<double> accumulator;

        //!A vector of vectors with the number of values accumulated in each correlator
        vector<vector< int> > nCorrelation;
        //!A vector of vectors with the actual correlation function
        vector<vector< double> > correlation;
        //!A vector of vector where incoming values get stored
        vector<vector< double> > shift;
    };
#endif
