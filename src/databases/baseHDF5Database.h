#ifndef BASEHDF5DATABASE_H
#define BASEHDF5DATABASE_H
#include "Simple2DCell.h"
#include "baseDatabase.h"
#include <unordered_set>
#include <vector>
#include <hdf5.h>

/*!
@class baseHDF5Database 

This class interfaces with the hdf5 c-api to provide a few common functions.
It is specialized to almost entirely deal with 2D data (i.e., datasets of some number of columns, for which each row is another "record") .

The constructor handles the correct file access as a function of the desired filemode (readonly, readwrite, or replace),
and a few functions of convenience are provided.
Small amounts of information can be added to the header of the hdf5 file by the addHeaderData function,
Datasets that can store an arbitrary number of records can be added by first
registering them (registerExtendableDataset(...)) and then writing the data (extendDataset(...)). 
The ability to read specific rows of datasets is also provided (readDataset(...)).
All functions are currently templated, with explicit instantiation for ints, floats, and doubles.

This class will mostly be used in the context of derived classes that, for instance, might set up the
reproducible structures to save snapshots with positions / velocities in a simulation, etc.
*/
class baseHDF5Database : public baseDatabaseInformation
    {
    public:
        //! The constructor takes the filename and mode, and correctly opens or creates the hdf5 file
        baseHDF5Database(std::string _filename, fileMode::Enum  _accessMode = fileMode::readonly);
        //! The destructor just closes the file
        ~baseHDF5Database();

        //! The file that will be accessed via the C-API of HDF5
        hid_t hdf5File;
        //! A record of all names registered as extendable datasets
        std::unordered_set<std::string> extendableDatasetNames;

        //! Add a small amount of data to the hdf5 header (best for at most a handful of elements)
        template<typename T>
        void addHeaderData(std::string name, const std::vector<T> &data);

        //! Set up a chunked dataset with a name that can have records of fixed maximum size added to it an unlimited number of times
        template<typename T>
        void registerExtendableDataset(std::string name,int maximumSizePerRecord);

        //! Take a vector of data and append it as a new row in the named dataset
        template<typename T>
        void extendDataset(std::string name,std::vector<T> &data);

        //! A helper function to get the number of records in a named dataset
        unsigned long getDatasetDimensions(std::string name);

        //! read a record of a named dataset. Default to the final row
        template<typename T>
        void readDataset(std::string name, std::vector<T> &readData, int record = -1);

        //! Return the data type to save a type as (e.g., H5T_NATIVE_INT for ints). Explicit instantiation in the cpp file
        template<typename T>
        hid_t getDatatypeFor();

        //! A function that tests functionality by calling several of the above functions
        void writeTest();

        //! A function that tests functionality by calling several of the above functions
        void readTest(int record);

        typedef shared_ptr<Simple2DCell> STATE;

        //!Write the current state; if the default value of rec=-1 is used, add a new entry
        virtual void writeState(STATE c, double time = -1.0, int rec = -1) {};
        //Read the rec state of the database. If geometry = true, call computeGeometry routines (instead of just reading in the d.o.f.s)
        virtual void readState(STATE c, int rec, bool geometry = true) {};
    };


#endif
