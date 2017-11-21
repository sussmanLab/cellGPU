#define ENABLE_CUDA
#include "vectorValueDatabase.h"
/*! \file vectorValueDatabase.cpp */

vectorValueDatabase::vectorValueDatabase(int vectorLength, string fn, NcFile::FileMode mode)
    :BaseDatabaseNetCDF(fn,mode)
    {
    N=vectorLength;
    switch(Mode)
        {
        case NcFile::ReadOnly:
            break;
        case NcFile::Write:
            GetDimVar();
            break;
        case NcFile::Replace:
            SetDimVar();
            break;
        case NcFile::New:
            SetDimVar();
            break;
        default:
            ;
        };
    };

void vectorValueDatabase::SetDimVar()
    {
    //Set the dimensions
    recDim = File.add_dim("rec");
    dofDim = File.add_dim("dof", N);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    vecVar = File.add_var("vector", ncDscalar,recDim, dofDim);
    valVar = File.add_var("value",ncDscalar,recDim);
    }

void vectorValueDatabase::GetDimVar()
    {
    //Get the dimensions
    recDim = File.get_dim("rec");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    vecVar = File.get_var("vector");
    valVar = File.get_var("value");
    }

void vectorValueDatabase::WriteState(vector<Dscalar> &vec,Dscalar val)
    {
    int rec = recDim->size();
    valVar->put_rec(&val,rec);
    vecVar->put_rec(&vec[0],rec);
    File.sync();
    };

