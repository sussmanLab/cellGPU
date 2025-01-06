#ifndef VECTORVALUEDATABASE_H
#define VECTORVALUEDATABASE_H

#include "baseHDF5Database.h"

/*!
@class valueVectorDatabase

This class uses the baseHDF5Database as a backend to write a hdf5 file with two unlimited records.
One corresponds to a single scalar, and the other to a vector of doubles,
and every call to the writeState function appends a record to each of these two datasets.

The usage might be something like
    int n;
    std::vector<double> vectorToWrite(n,0);
    double value = 0;
    valueVectorDatabase stateSaver("./data/test.h5",vectorToWrite.size(),fileMode::replace);
    //other code to populate the vector and value
    stateSaver.writeState(vectorToWrite,value);
    //more code to repopulate the vector and value
    stateSaver.writeState(vectorToWrite,value);
    //etc.
*/
class valueVectorDatabase : public baseHDF5Database
    {
    public:
        //!The constructor calls the baseHDF5Database constructor (to handle fileMode stuff), sets data structures, and registers the datasets in the hdf5 file if needed
        valueVectorDatabase(std::string _filename, unsigned long vectorSize, fileMode::Enum _accessMode = fileMode::readonly);

        //! create the two unlimited datasets, "/vector" and "/value", in the hdf5 file
        void registerDatasets();
        //! return the number of records in the dataset
        unsigned long currentNumberOfRecords();
        //! Append the data passed to this function as a new record in the file
        void writeState(double val, std::vector<double> &data);
        //! populate valueVector[0] and dataVector with the values in the corresponding data rows
        void readState(int record);

        unsigned long maximumVectorSize;
        //! the zeroth element will be populated after a readState call
        std::vector<double> valueVector;
        //! will be populated after a readState call
        std::vector<double> dataVector;
    };

#endif
