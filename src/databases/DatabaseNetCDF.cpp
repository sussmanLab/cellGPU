#define ENABLE_CUDA
#include "DatabaseNetCDF.h"
/*! \file DatabaseNetCDF.cpp */

BaseDatabaseNetCDF::BaseDatabaseNetCDF(string fn, NcFile::FileMode mode)
     : BaseDatabase(fn,mode),
     File(fn.c_str(), mode)
{
    NcError err(NcError::silent_nonfatal);
}


