#ifndef DATABASE_AVM_H
#define DATABASE_AVM_H

#include "std_include.h"
#include "avm2d.h"
#include "spv2d.h"
#include "Database.h"
#include <netcdfcpp.h>
#include <string>
#include "vector_types.h"


/*!
Class for a state database for an active vertex model state
the box dimensions are stored, the 2d unwrapped coordinate of vertices,
and the set of connections between vertices
*/
//!Simple databse for reading/writing 2d AVM states
class AVMDatabase : public BaseDatabase
{
private:
    typedef AVM2D STATE;
    int Nv; //!< number of vertices in AVM
    int Nc; //!< number of cells in AVM
    int Nvn; //!< the number of vertex-vertex connections
    NcDim *recDim, *NvDim, *dofDim, *NvnDim, *ncDim, *boxDim, *unitDim; //!< NcDims we'll use
    NcVar *posVar, *forceVar, *vneighVar, *directorVar, *BoxMatrixVar, *timeVar; //!<NcVars we'll use
    int Current;    //!< keeps track of the current record when in write mode


public:
    //!The default constructor takes the number of *vertices* as the parameter
    AVMDatabase(int nv, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
    ~AVMDatabase(){File.close();};

private:
    void SetDimVar();
    void GetDimVar();

public:
    int  GetCurrentRec(); //!<Return the current record of the database
    //!Get the total number of records in the database
    int GetNumRecs()
        {
        recDim = File.get_dim("rec");
        return recDim->size();
        };

    //!Write the current state of the system to the database. If the default value of "rec=-1" is used, just append the current state to a new record at the end of the database
    void WriteState(STATE &c, Dscalar time = -1.0, int rec=-1);
    //!Read the "rec"th entry of the database into AVM2D state c. If geometry=true, after reading a CPU-based triangulation is performed, and local geometry of cells computed.
    void ReadState(STATE &c, int rec,bool geometry=true);
};

//Implementation


AVMDatabase::AVMDatabase(int np, string fn, NcFile::FileMode mode)
    : BaseDatabase(fn,mode),
      Nv(np),
      Current(0)
{
    Nc = np/2;
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

void AVMDatabase::SetDimVar()
{
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    ncDim  = File.add_dim("Nc",  Nc);
    dofDim  = File.add_dim("dof",  Nv*2);
    NvnDim = File.add_dim("Nvn", Nv*3);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    posVar          = File.add_var("pos",       ncDscalar,recDim, dofDim);
    forceVar          = File.add_var("force",       ncDscalar,recDim, dofDim);
    vneighVar          = File.add_var("Vneighs",         ncInt,recDim, NvnDim );
    directorVar          = File.add_var("director",         ncDscalar,recDim, ncDim );
    BoxMatrixVar    = File.add_var("BoxMatrix", ncDscalar,recDim, boxDim);
    timeVar          = File.add_var("time",     ncDscalar,recDim, unitDim);
}

void AVMDatabase::GetDimVar()
{
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim  = File.get_dim("dof");
    NvnDim = File.get_dim("Nvn");
    unitDim = File.get_dim("unit");
    //Get the variables
    posVar          = File.get_var("pos");
    forceVar          = File.get_var("force");
    vneighVar          = File.get_var("Vneighs");
    directorVar          = File.get_var("director");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    timeVar    = File.get_var("time");
}


void AVMDatabase::WriteState(STATE &s, Dscalar time, int rec)
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
    std::vector<Dscalar> forcedat(2*Nv);
    std::vector<Dscalar> directordat(Nc);
    std::vector<int> vndat(3*Nv);
    int idx = 0;

    ArrayHandle<Dscalar2> h_p(s.vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(s.vertexForces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s.cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn(s.vertexNeighbors,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s.tagToIdx[ii];
        directordat[ii] = h_cd.data[pidx];
        };
    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s.tagToIdxVertex[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        Dscalar fx = h_f.data[pidx].x;
        Dscalar fy = h_f.data[pidx].y;
        forcedat[(2*idx)] = fx;
        forcedat[(2*idx)+1] = fy;
        idx +=1;
        };
    for (int vv = 0; vv < Nv; ++vv)
        {
        int vertexIndex = s.tagToIdxVertex[vv];
        for (int ii = 0 ;ii < 3; ++ii)
            {
            vndat[3*vv+ii] = s.idxToTagVertex[h_vn.data[3*vertexIndex+ii]];
            };
        };
    /*!
     * \todo once hilbert sorting is working for vertex models, make sure database saving is correct
     */

    //Write all the data
    timeVar      ->put_rec(&time,      rec);
    posVar      ->put_rec(&posdat[0],     rec);
    forceVar      ->put_rec(&forcedat[0],     rec);
    vneighVar       ->put_rec(&vndat[0],      rec);
    directorVar       ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);

    File.sync();
}

#endif
