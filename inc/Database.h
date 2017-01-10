#ifndef DATABASE_H
#define DATABASE_H


#include "std_include.h"
#include <netcdfcpp.h>
#include <string>
#include "vector_types.h"


using namespace std;
//!A base class that controls a netCDF-based data implementation
class BaseDatabase
{
public:
    //! The name of the file
    string filename;
    //!The desired netCDF mode (replace, new, readonly, etc.)
    const int Mode;
    //!The NcFile itself
    NcFile File;

    //!The default constructor takes starts a bland filename in readonly mode
    BaseDatabase(string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
};

BaseDatabase::BaseDatabase(string fn, NcFile::FileMode mode)
    : filename(fn),
      Mode(mode),
      File(fn.c_str(), mode)
{
    NcError err(NcError::silent_nonfatal);
}

#endif
