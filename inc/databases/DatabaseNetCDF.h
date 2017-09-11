#ifndef DATABASE_H
#define DATABASE_H

#include <netcdfcpp.h>
#include "BaseDatabase.h"

/*! \file DatabaseNetCDF.h */
//! A base class that implements a details-free  netCDF4-based data storage system
/*!
BaseDatabase just provides an interface to a file and a mode of operation.
*/
class BaseDatabaseNetCDF : public BaseDatabase
    {
    public:
        //!The NcFile itself
        NcFile File;

        //!The default constructor starts a bland filename in readonly mode
        BaseDatabaseNetCDF(string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
    };

#endif
