#include "nvtModelDatabase.h"
/*! \file nvtModelDatabase.cpp */

nvtModelDatabase::nvtModelDatabase(int np, string fn, NcFile::FileMode mode)
    : BaseDatabaseNetCDF(fn,mode),
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

void nvtModelDatabase::SetDimVar()
    {
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    posVar              = File.add_var("position",       ncDouble,recDim, dofDim);
    velVar              = File.add_var("velocity",      ncDouble,recDim, dofDim);
    typeVar             = File.add_var("type",          ncInt,recDim, NvDim );
    additionalDataVar   = File.add_var("additionalData",ncDouble,recDim,dofDim);
    BoxMatrixVar        = File.add_var("BoxMatrix",     ncDouble,recDim, boxDim);
    timeVar             = File.add_var("time",     ncDouble,recDim, unitDim);
    }

void nvtModelDatabase::GetDimVar()
    {
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    posVar = File.get_var("position");
    velVar = File.get_var("velocity");
    typeVar = File.get_var("type");
    additionalDataVar = File.get_var("additionalData");
    BoxMatrixVar = File.get_var("BoxMatrix");
    timeVar = File.get_var("time");

    }

void nvtModelDatabase::writeState(STATE c, double time, int rec)
    {
    shared_ptr<VoronoiQuadraticEnergy> s = dynamic_pointer_cast<VoronoiQuadraticEnergy>(c);
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<double> boxdat(4,0.0);
    double x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<double> posdat(2*Nv);
    std::vector<double> veldat(2*Nv);
    std::vector<double> additionaldat(2*Nv);
    std::vector<int> typedat(Nv);
    int idx = 0;

    ArrayHandle<double2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(s->returnVelocities());
    ArrayHandle<double2> h_m(s->returnAreaPeriPreferences());
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        double px = h_p.data[pidx].x;
        double py = h_p.data[pidx].y;
        double vx = h_v.data[pidx].x;
        double vy = h_v.data[pidx].y;
        double ma = h_m.data[pidx].x;
        double mb = h_m.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        veldat[(2*idx)] = vx;
        veldat[(2*idx)+1] = vy;
        additionaldat[(2*idx)] = ma;
        additionaldat[(2*idx)+1] = mb;
        typedat[ii] = h_ct.data[pidx];
        idx +=1;
        };

    //Write all the data
    posVar           ->put_rec(&posdat[0],       rec);
    velVar           ->put_rec(&veldat[0],       rec);
    additionalDataVar->put_rec(&additionaldat[0],rec);
    typeVar          ->put_rec(&typedat[0],      rec);
    timeVar          ->put_rec(&time,            rec);
    BoxMatrixVar     ->put_rec(&boxdat[0],       rec);
    File.sync();
    }

void nvtModelDatabase::readState(STATE c, int rec,bool geometry)
    {
    shared_ptr<VoronoiQuadraticEnergy> t = dynamic_pointer_cast<VoronoiQuadraticEnergy>(c);
    //initialize the NetCDF dimensions and variables
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(& t->currentTime,1,1);

    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<double> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t->Box->setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions and velocities
    posVar-> set_cur(rec);
    velVar-> set_cur(rec);
    additionalDataVar-> set_cur(rec);
    std::vector<double> posdata(2*Nv,0.0);
    std::vector<double> veldata(2*Nv,0.0);
    std::vector<double> additionaldata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());
    velVar->get(&veldata[0],1, dofDim->size());
    additionalDataVar->get(&additionaldata[0],1,dofDim->size());

    ArrayHandle<double2> h_p(t->cellPositions,access_location::host,access_mode::overwrite);
    ArrayHandle<double2> h_v(t->returnVelocities(),access_location::host,access_mode::overwrite);
    ArrayHandle<double2> h_m(t->returnAreaPeriPreferences(),access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        double px = posdata[(2*idx)];
        double py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        double vx = veldata[(2*idx)];
        double vy = veldata[(2*idx)+1];
        h_v.data[idx].x=vx;
        h_v.data[idx].y=vy;
        double ma = additionaldata[(2*idx)];
        double mb = additionaldata[(2*idx)+1];
        h_m.data[idx].x=ma;
        h_m.data[idx].y=mb;
        };

    //get cell types and cell directors
    typeVar->set_cur(rec);
    std::vector<int> ctdata(Nv,0.0);
    typeVar->get(&ctdata[0],1, NvDim->size());
    ArrayHandle<int> h_ct(t->cellType,access_location::host,access_mode::overwrite);

    for (int idx = 0; idx < Nv; ++idx)
        {
        h_ct.data[idx]=ctdata[idx];;
        };

    //by default, compute the triangulation and geometrical information
    if(geometry)
        {
        t->globalTriangulationDelGPU();
        t->resetLists();
        if(t->GPUcompute)
            t->computeGeometryGPU();
        else
            t->computeGeometryCPU();
        };
    }


