#ifndef DATABASE_H
#define DATABASE_H


#include "std_include.h"
#include <netcdfcpp.h>
#include <string>
#include "vector_types.h"


using namespace std;
class BaseDatabase
{
public:
    string filename;
    const int Mode;
    NcFile File;

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


/////////////////////////////////////////////////////////////////////////////////
//class for a state database for a 2d delaunay triangulation
//the box dimensions are stored, the 2d unwrapped coordinate of the delaunay vertices,
//and the shape index parameter for each vertex
/////////////////////////////////////////////////////////////////////////////////

class SPVDatabase : public BaseDatabase
{
private:
    typedef SPV2D STATE;
    int Nv; // number of vertices in delaunay triangulation
    NcDim *recDim, *NvDim, *dofDim, *boxDim, *unitDim;
    NcVar *posVar, *typeVar, *directorVar, *BoxMatrixVar, *timeVar, *means0Var,*exVar;
    bool exclusions;
    int Current;


public:
    SPVDatabase(int np, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly,bool excluded = false);
    ~SPVDatabase(){File.close();};

private:
    void SetDimVar();
    void GetDimVar();

public:
    void SetCurrentRec(int r);
    int  GetCurrentRec();
    int GetNumRecs(){
                    NcDim *rd = File.get_dim("rec");
                    return rd->size();
                    };

    void WriteState(STATE &c, Dscalar time = -1.0, int rec=-1);
    void ReadState(STATE &c, int rec,bool geometry=true);
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   IMPLEMENTATION   ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

SPVDatabase::SPVDatabase(int np, string fn, NcFile::FileMode mode, bool exclude)
    : BaseDatabase(fn,mode),
      Nv(np),
      Current(0),
      exclusions(exclude)
{
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
}

void SPVDatabase::SetDimVar()
{
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    timeVar          = File.add_var("time",     ncDscalar,recDim, unitDim);
    means0Var          = File.add_var("means0",     ncDscalar,recDim, unitDim);
    posVar          = File.add_var("pos",       ncDscalar,recDim, dofDim);
    typeVar          = File.add_var("type",         ncInt,recDim, NvDim );
    directorVar          = File.add_var("director",         ncDscalar,recDim, NvDim );
    BoxMatrixVar    = File.add_var("BoxMatrix", ncDscalar,recDim, boxDim);
    if(exclusions)
        exVar          = File.add_var("externalForce",       ncDscalar,recDim, dofDim);
}

void SPVDatabase::GetDimVar()
{
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    posVar          = File.get_var("pos");
    typeVar          = File.get_var("type");
    directorVar          = File.get_var("director");
    means0Var          = File.get_var("means0");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    timeVar    = File.get_var("time");
    if(exclusions)
        exVar = File.get_var("externalForce");
}

void SPVDatabase::SetCurrentRec(int r)
{
    Current = r;
}

int SPVDatabase::GetCurrentRec()
{
    return Current;
}

void SPVDatabase::WriteState(STATE &s, Dscalar time, int rec)
{
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s.Timestep*s.deltaT;

    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s.Box.getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> directordat(Nv);
    std::vector<int> typedat(Nv);
    int idx = 0;
    Dscalar means0=0.0;

    ArrayHandle<Dscalar2> h_p(s.points,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s.cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s.CellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_ex(s.exclusions,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s.tagToIdx[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        directordat[ii] = h_cd.data[pidx];
        if(h_ex.data[ii] == 0)
            typedat[ii] = h_ct.data[pidx];
        else
            typedat[ii] = h_ct.data[pidx]-5;
        idx +=1;
        };
//    means0 = means0/Nv;
    means0 = s.reportq();

    //Write all the data
    means0Var      ->put_rec(&means0,      rec);
    timeVar      ->put_rec(&time,      rec);
    posVar      ->put_rec(&posdat[0],     rec);
    typeVar       ->put_rec(&typedat[0],      rec);
    directorVar       ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);
    if(exclusions)
        {
        ArrayHandle<Dscalar2> h_ef(s.external_forces,access_location::host,access_mode::read);
        std::vector<Dscalar> exdat(2*Nv);
        int id = 0;
        for (int ii = 0; ii < Nv; ++ii)
            {
            int pidx = s.tagToIdx[ii];
            Dscalar px = h_ef.data[pidx].x;
            Dscalar py = h_ef.data[pidx].y;
            exdat[(2*id)] = px;
            exdat[(2*id)+1] = py;
            id +=1;
            };
        exVar      ->put_rec(&exdat[0],     rec);
        };

    File.sync();
}

//overwrites a tissue that needs to have the correct number of cells when passed to this function
void SPVDatabase::ReadState(STATE &t, int rec,bool geometry)
{
    //initialize the NetCDF dimensions and variables
    //test if there is exclusion data to read...
    int tester = File.num_vars();
    if (tester == 7)
        exclusions = true;
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(&SimTime,1);


    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<Dscalar> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t.Box.setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions
    posVar-> set_cur(rec);
    std::vector<Dscalar> posdata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());

    ArrayHandle<Dscalar2> h_p(t.points,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        Dscalar px = posdata[(2*idx)];
        Dscalar py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };

    //get cell types and cell directors
    typeVar->set_cur(rec);
    std::vector<int> ctdata(Nv,0.0);
    typeVar->get(&ctdata[0],1, NvDim->size());
    ArrayHandle<int> h_ct(t.CellType,access_location::host,access_mode::overwrite);

    directorVar->set_cur(rec);
    std::vector<Dscalar> cddata(Nv,0.0);
    directorVar->get(&cddata[0],1, NvDim->size());
    ArrayHandle<Dscalar> h_cd(t.cellDirectors,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        h_cd.data[idx]=cddata[idx];;
        h_ct.data[idx]=ctdata[idx];;
        };

    //read in excluded forces if applicable...
    if (tester == 7)
        {
        exVar-> set_cur(rec);
        std::vector<Dscalar> efdata(2*Nv,0.0);
        exVar->get(&posdata[0],1, dofDim->size());
        ArrayHandle<Dscalar2> h_ef(t.external_forces,access_location::host,access_mode::overwrite);
        for (int idx = 0; idx < Nv; ++idx)
            {
            Dscalar efx = efdata[(2*idx)];
            Dscalar efy = efdata[(2*idx)+1];
            h_ef.data[idx].x=efx;
            h_ef.data[idx].y=efy;
            };
        };


    //by default, compute the triangulation and geometrical information
    if(geometry)
        {
        t.resetDelLocPoints();
        t.updateCellList();
        t.globalTriangulationCGAL();
        if(t.GPUcompute)
            t.computeGeometryGPU();
        else
            t.computeGeometryCPU();
        };

}



#endif
