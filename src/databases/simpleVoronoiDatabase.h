#ifndef SIMPLEVORONOIDATABASE_H
#define SIMPLEVORONOIDATABASE_H

#include "baseHDF5Database.h"
#include "voronoiQuadraticEnergy.h"

/*! \file simpleVoronoiDatabase.h */
//!Simple databse for reading/writing 2d spv states
/*!
Class for a state database for a 2d delaunay triangulation
the box dimensions are stored, the 2d unwrapped coordinate of the delaunay vertices,
and the shape index parameter for each vertex
*/
class simpleVoronoiDatabase : public baseHDF5Database
{
public:
    simpleVoronoiDatabase(int np, string fn="temp.nc",fileMode::Enum _mode=fileMode::readonly);

public:

    //!Write the current state of the system to the database. If the default value of "rec=-1" is used, just append the current state to a new record at the end of the database
    virtual void writeState(STATE c, double time = -1.0, int rec=-1);
    //!Read the "rec"th entry of the database into SPV2D state c. If geometry=true, the local geometry of cells computed (so that further simulations can be run); set to false if you just want to load and analyze configuration data.
    virtual void readState(STATE c, int rec,bool geometry=true);

private:
    typedef shared_ptr<Simple2DCell> STATE;
    int N; //!< number of points
    int Current;    //!< keeps track of the current record when in write mode

    //! a vector of length 1
    std::vector<double> timeVector;
    //! a vector of length 4
    std::vector<double> boxVector;
    //! a vector of length 2*N
    std::vector<double> coordinateVector;
    //! a vector of length N
    std::vector<double> doubleVector;
    //! a vector of length N
    std::vector<int> intVector;
    //! The number of frames that have been saved so far
    unsigned long currentNumberOfRecords();

    void registerDatasets();

};
#endif
