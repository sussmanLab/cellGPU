#ifndef DATABASE_H
#define DATABASE_H


#include <netcdfcpp.h>
#include <string>
#include "vector_types.h"
#include "spv2d.h"


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

class TriangulationDatabase : public BaseDatabase
{
private:
    typedef DelaunayMD STATE;
    int Nv; // number of vertices in delaunay triangulation
    NcDim *recDim, *NvDim, *dofDim, *boxDim, *unitDim;
    NcVar *posVar, *s0Var, *BoxMatrixVar, *timeVar, *means0Var;

    int Current;


public:
    TriangulationDatabase(int np, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);

private:
    void SetDimVar();
    void GetDimVar();

public:
    void SetCurrentRec(int r);
    int  GetCurrentRec();

    void WriteState(STATE const &c, double time = -1.0, int rec=-1);
    void ReadState(STATE &c, int rec);
    void ReadNextState(STATE &c);
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   IMPLEMENTATION   ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

TriangulationDatabase::TriangulationDatabase(int np, string fn, NcFile::FileMode mode)
    : BaseDatabase(fn,mode),
      Nv(np),
      Current(0)
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

void TriangulationDatabase::SetDimVar()
{
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    timeVar          = File.add_var("time",     ncDouble,recDim, unitDim);
    means0Var          = File.add_var("means0",     ncDouble,recDim, unitDim);
    posVar          = File.add_var("pos",       ncDouble,recDim, dofDim);
    s0Var          = File.add_var("s0",         ncDouble,recDim, NvDim );
    BoxMatrixVar    = File.add_var("BoxMatrix", ncDouble,recDim, boxDim);
}

void TriangulationDatabase::GetDimVar()
{
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    posVar          = File.get_var("pos");
    s0Var          = File.get_var("s0");
    means0Var          = File.get_var("means0");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    timeVar    = File.get_var("time");
}

void TriangulationDatabase::SetCurrentRec(int r)
{
    Current = r;
}

int TriangulationDatabase::GetCurrentRec()
{
    return Current;
}

void TriangulationDatabase::WriteState(STATE const &s, double time, int rec)
{
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = rec;

    std::vector<double> boxdat(4,0.0);
//    boxdat[0] = s._box.xDimension();
//    boxdat[3] = s._box.yDimension();
    std::vector<double> posdat(2*Nv);
    std::vector<double> s0dat(Nv);
    int idx = 0;
    double means0=0.0;

    ArrayHandle<float2> h_p(s.points,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
        {
//        double px = c->positionNotInBox().x();
//        double py = c->positionNotInBox().y();
//        double s0 = (c->perimeter()) / sqrt(c->area());
        double px = h_p.data[ii].x;
        double py = h_p.data[ii].y;
        double s0 = 0.0;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        s0dat[idx] = s0;
        means0+=s0;
        idx +=1;
        };
    means0 = means0/Nv;

    //Write all the data
    means0Var      ->put_rec(&means0,      rec);
    timeVar      ->put_rec(&time,      rec);
    posVar      ->put_rec(&posdat[0],     rec);
    s0Var       ->put_rec(&s0dat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);
    File.sync();
}

//overwrites a tissue that needs to have the correct number of cells when passed to this function
void TriangulationDatabase::ReadState(STATE &t, int rec)
{
    GetDimVar();

    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<double> boxdata(4,0.0);
//    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
//    t._box.setBox(boxdata[0],boxdata[3]);

    //get the positions
    posVar-> set_cur(rec);
    std::vector<double> posdata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());
    int idx = 0;
    ArrayHandle<float2> h_p(t.points,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        double px = posdata[(2*idx)];
        double py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };
    t.resetDelLocPoints();
    t.updateCellList();

}

/*





*/


#endif
