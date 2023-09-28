#include "vectorValueDatabase.h"
/*! \file vectorValueDatabase.cpp */

vectorValueDatabase::vectorValueDatabase(int vectorLength, string fn, NcFile::FileMode mode)
    :BaseDatabaseNetCDF(fn,mode)
    {
    N=vectorLength;
    val=0.0;
    vec.resize(N);
    switch(Mode)
        {
        case NcFile::ReadOnly:
            GetDimVar();
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
    vecVar = File.add_var("vector", ncDouble,recDim, dofDim);
    valVar = File.add_var("value",ncDouble,recDim);
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

void vectorValueDatabase::writeState(vector<double> &vec,double val)
    {
    int rec = recDim->size();
    valVar->put_rec(&val,rec);
    vecVar->put_rec(&vec[0],rec);
    File.sync();
    };

void vectorValueDatabase::readState(int rec)
    {
    int totalRecords = GetNumRecs();
    if (rec >= totalRecords)
        {
        printf("Trying to read a database entry that does not exist\n");
        throw std::exception();
        };
        vecVar->set_cur(rec);
        vecVar->get(&vec[0],1,dofDim->size());
        valVar->set_cur(rec);
        valVar->get(&val,1,1);
    };
