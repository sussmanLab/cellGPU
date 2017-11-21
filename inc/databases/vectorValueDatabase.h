#ifndef vectorValueDatabase_h
#define vectorValueDatabase_h

#include "DatabaseNetCDF.h"

/*! \file vectorValueDatabase.h */
//! Sometimes it is convenient to have a write-only NetCDF database where every record is a value and a vector
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
        //!write a new value and vector
        void WriteState(vector<Dscalar> &vec,Dscalar val);

    protected:
        void SetDimVar();
        void GetDimVar();
        int N;
    };


#endif
