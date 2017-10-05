#define ENABLE_CUDA
#include "DatabaseNetCDFAVM.h"
/*! \file DatabaseNetCDFAVM.cpp */

/*! Base constructor implementation */
AVMDatabaseNetCDF::AVMDatabaseNetCDF(int np, string fn, NcFile::FileMode mode)
    : BaseDatabaseNetCDF(fn,mode),
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

void AVMDatabaseNetCDF::SetDimVar()
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

void AVMDatabaseNetCDF::GetDimVar()
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

void AVMDatabaseNetCDF::ReadState(STATE t, int rec, bool geometry)
    {
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(& t->currentTime,1,1);
    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<Dscalar> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t->Box->setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions
    posVar-> set_cur(rec);
    std::vector<Dscalar> posdata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());

    ArrayHandle<Dscalar2> h_p(t->vertexPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        Dscalar px = posdata[(2*idx)];
        Dscalar py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };
    
    //set the vertex neighbors
    ArrayHandle<int> h_vn(t->vertexNeighbors,access_location::host,access_mode::read);
    vneighVar->set_cur(rec);
    std::vector<Dscalar> vndat(3*Nv,0.0);
    vneighVar       ->get(&vndat[0],1,NvnDim->size());
    for (int vv = 0; vv < Nv; ++vv)
        {
        for (int ii = 0 ;ii < 3; ++ii)
            {
            h_vn.data[3*vv+ii] = vndat[3*vv+ii];
            };
        };
    //use this to reconstruct network topology
    //
    if (geometry)
        {
        t->computeGeometryCPU();
        };
    };


void AVMDatabaseNetCDF::WriteState(STATE s, Dscalar time, int rec)
{
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> forcedat(2*Nv);
    std::vector<Dscalar> directordat(Nc);
    std::vector<int> vndat(3*Nv);
    int idx = 0;

    ArrayHandle<Dscalar2> h_p(s->vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(s->vertexForces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn(s->vertexNeighbors,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        directordat[ii] = h_cd.data[pidx];
        };
    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdxVertex[ii];
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
        int vertexIndex = s->tagToIdxVertex[vv];
        for (int ii = 0 ;ii < 3; ++ii)
            {
            vndat[3*vv+ii] = s->idxToTagVertex[h_vn.data[3*vertexIndex+ii]];
            };
        };

    //Write all the data
    timeVar      ->put_rec(&time,      rec);
    posVar      ->put_rec(&posdat[0],     rec);
    forceVar      ->put_rec(&forcedat[0],     rec);
    vneighVar       ->put_rec(&vndat[0],      rec);
    directorVar       ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);

    File.sync();
}
