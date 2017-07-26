#ifndef DATABASE_H
#define DATABASE_H

#include "std_include.h"
#include <netcdfcpp.h>
#include <string>
#include "vector_types.h"

/*! \file DatabaseNetCDF.h */
//! A base class that implements a details-free  netCDF4-based data storage system
/*!
BaseDatabase just provides an interface to a file and a mode of operation.
*/
class BaseDatabaseNetCDF
{
public:
    //! The name of the file
    string filename;
    //!The desired netCDF mode (replace, new, readonly, etc.)
    const int Mode;
    //!The NcFile itself
    NcFile File;

    //!The default constructor takes starts a bland filename in readonly mode
    BaseDatabaseNetCDF(string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
};

BaseDatabaseNetCDF::BaseDatabaseNetCDF(string fn, NcFile::FileMode mode)
    : filename(fn),
      Mode(mode),
      File(fn.c_str(), mode)
{
    NcError err(NcError::silent_nonfatal);
}

#endif
