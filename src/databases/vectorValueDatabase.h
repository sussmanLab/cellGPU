#ifndef vectorValueDatabase_h
#define vectorValueDatabase_h

#include "DatabaseNetCDF.h"

/*! \file vectorValueDatabase.h */
//! Sometimes it is convenient to have a NetCDF database where every record is a value and a vector
/*!
There is one unlimited dimension, and each record stores a scalar value and a vector of scalars
*/
class vectorValueDatabase : public BaseDatabaseNetCDF
    {
    public:
        vectorValueDatabase(int vectorLength, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
        ~vectorValueDatabase(){File.close();};

        //! NcDims we'll use
        NcDim *recDim, *dofDim, *unitDim;
        //! NcVars
        NcVar *valVar, *vecVar;
        //!read values in a new value and vector
        void readState(int rec);
        //!write a new value and vector
        void writeState(vector<double> &vec,double val);
        //!read the number of records in the database
        int GetNumRecs(){
                    NcDim *rd = File.get_dim("rec");
                    return rd->size();
                    };
        //!The variable that will be loaded for "value" when state is read
        double val;
        //!The variable that will be loaded for "vector" when state is read
        vector<double> vec;

    protected:
        void SetDimVar();
        void GetDimVar();
        //!Length of the vector
        int N;
    };
#endif
