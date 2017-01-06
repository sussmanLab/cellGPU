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
    string filename; //!< The name of the file
    const int Mode;  //!< The desired netCDF mode (repalce, new, readonly, etc.)
    NcFile File;    //!< The NcFile itself

    //!The default constructor takes starts a bland filename in readonly mode
    BaseDatabase(string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   IMPLEMENTATION   ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

BaseDatabase::BaseDatabase(string fn, NcFile::FileMode mode)
    : filename(fn),
      Mode(mode),
      File(fn.c_str(), mode)
{
    NcError err(NcError::silent_nonfatal);
}

#endif
