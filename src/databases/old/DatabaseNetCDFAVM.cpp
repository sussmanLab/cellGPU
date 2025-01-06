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
    recDim  = File.add_dim("rec");
    NvDim   = File.add_dim("Nv",  Nv);
    ncDim   = File.add_dim("Nc",  Nc);
    nc2Dim  = File.add_dim("Nc2", 2* Nc);
    dofDim  = File.add_dim("dof",  Nv*2);
    NvnDim  = File.add_dim("Nvn", Nv*3);
    boxDim  = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    posVar       = File.add_var("pos",       ncDouble,recDim, dofDim);
    forceVar     = File.add_var("force",       ncDouble,recDim, dofDim);
    vcneighVar   = File.add_var("VertexCellNeighbors",         ncInt,recDim, NvnDim );
    vneighVar    = File.add_var("Vneighs",         ncInt,recDim, NvnDim );
    cellTypeVar  = File.add_var("cellType",         ncDouble,recDim, ncDim );
    directorVar  = File.add_var("director",         ncDouble,recDim, ncDim );
    cellPosVar   = File.add_var("cellPositions",         ncDouble,recDim, nc2Dim );
    BoxMatrixVar = File.add_var("BoxMatrix", ncDouble,recDim, boxDim);
    meanqVar     = File.add_var("meanQ",     ncDouble,recDim, unitDim);
    timeVar      = File.add_var("time",     ncDouble,recDim, unitDim);
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
    vcneighVar          = File.get_var("VertexCellNeighbors");
    directorVar          = File.get_var("director");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    meanqVar  = File.get_var("meanQ");
    timeVar    = File.get_var("time");
}

/*!
Vertex model reading not currently functional
*/
void AVMDatabaseNetCDF::readState(STATE t, int rec, bool geometry)
    {
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(& t->currentTime,1,1);
    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<double> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t->Box->setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions
    posVar-> set_cur(rec);
    std::vector<double> posdata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());

    ArrayHandle<double2> h_p(t->vertexPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        double px = posdata[(2*idx)];
        double py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };

    //set the vertex neighbors and vertex-cell neighbors
    ArrayHandle<int> h_vn(t->vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(t->vertexCellNeighbors,access_location::host,access_mode::read);
    vneighVar->set_cur(rec);
    vcneighVar->set_cur(rec);
    std::vector<double> vndat(3*Nv,0.0);
    std::vector<double> vcndat(3*Nv,0.0);
    vneighVar       ->get(&vndat[0],1,NvnDim->size());
    vcneighVar       ->get(&vcndat[0],1,NvnDim->size());
    for (int vv = 0; vv < Nv; ++vv)
        {
        for (int ii = 0 ;ii < 3; ++ii)
            {
            h_vn.data[3*vv+ii] = vndat[3*vv+ii];
            h_vcn.data[3*vv+ii] = vcndat[3*vv+ii];
            };
        };
    //use this to reconstruct network topology
    //first just get all of the cell vertices; we'll order them later
    int Nc = Nv/2;
    vector<int> cvn(Nc,0);
    for (int vv = 0; vv <Nv; ++vv)
        for (int ii = 0; ii < 3; ++ii)
            {
            int cell =  vcndat[3*vv+ii];
            cvn[cell] +=1;
            };
    int nMax = 0;
    ArrayHandle<int> h_nn(t->cellVertexNum);
    for (int cc = 0; cc < Nc; ++cc)
        {
        h_nn.data[cc] = cvn[cc];
        if (cvn[cc] > nMax)
            nMax = cvn[cc];
        cvn[cc]=0;
        };
    t->vertexMax = nMax+2;
    t->cellVertices.resize((nMax+2)*Nc);
    t->n_idx = Index2D(nMax+2,Nc);

    ArrayHandle<int> h_n(t->cellVertices);
    for (int vv = 0; vv <Nv; ++vv)
        for (int ii = 0; ii < 3; ++ii)
            {
            int cell =  vcndat[3*vv+ii];
            h_n.data[t->n_idx(cvn[cell],cell)] = vv;
            cvn[cell] +=1;
            };

    //now put all of those vertices in ccw order
    for (int cc = 0; cc < Nc; ++cc)
        {
        int neigh = h_nn.data[cc];
        //The voronoi vertices, relative to cell CM
        vector< double2> Vpoints(neigh);
        int v0 =  h_n.data[t->n_idx(0,cc)];
        double2 meanPos = make_double2(0.0,0.0);
        vector<int> originalVertexOrder(neigh);
        for (int vv = 0; vv < neigh; ++vv)
            {
            int v1 =  h_n.data[t->n_idx(vv,cc)];
            originalVertexOrder[vv]=v1;
            t->Box->minDist(h_p.data[v0],h_p.data[v1],Vpoints[vv]);
            meanPos = meanPos + Vpoints[vv];
            };
        for (int vv = 0; vv < neigh; ++vv)
            Vpoints[vv] = Vpoints[vv] - meanPos;
        //a structure to sort them
        vector<pair <double,int> > CCWorder(neigh);
        for (int vv = 0; vv < neigh; ++vv)
            {
            CCWorder[vv].first = atan2(Vpoints[vv].y,Vpoints[vv].x);
            CCWorder[vv].second = vv;
            };
        sort(CCWorder.begin(),CCWorder.begin()+neigh);
        for (int vv = 0; vv < neigh; ++vv)
            {
            int orderedVertexIndex = CCWorder[neigh-1-vv].second;
            h_n.data[t->n_idx(vv,cc)] = originalVertexOrder[orderedVertexIndex];
            }
        };

    if (geometry)
        {
        t->computeGeometryCPU();
        };
    };


void AVMDatabaseNetCDF::writeState(STATE s, double time, int rec)
{
    Records +=1;
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
    std::vector<double> forcedat(2*Nv);
    std::vector<double> directordat(Nc);
    std::vector<int> typedat(Nc);
    std::vector<int> vndat(3*Nv);
    std::vector<int> vcndat(3*Nv);
    int idx = 0;

    ArrayHandle<double2> h_p(s->vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<double2> h_f(s->vertexForces,access_location::host,access_mode::read);
    ArrayHandle<double> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn(s->vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(s->vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);

    std::vector<double> cellPosDat(2*Nc);
    s->getCellPositionsCPU();
    ArrayHandle<double2> h_cpos(s->cellPositions);
    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        directordat[ii] = h_cd.data[pidx];
        typedat[ii] = h_ct.data[pidx];
        cellPosDat[2*ii+0] = h_cpos.data[pidx].x;
        cellPosDat[2*ii+1] = h_cpos.data[pidx].y;
        };
    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdxVertex[ii];
        double px = h_p.data[pidx].x;
        double py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        double fx = h_f.data[pidx].x;
        double fy = h_f.data[pidx].y;
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
            vcndat[3*vv+ii] = s->idxToTagVertex[h_vcn.data[3*vertexIndex+ii]];
            };
        };

    double meanq = s->reportq();
    //Write all the data
    timeVar     ->put_rec(&time,      rec);
    meanqVar    ->put_rec(&meanq,rec);
    posVar      ->put_rec(&posdat[0],     rec);
    forceVar    ->put_rec(&forcedat[0],     rec);
    vneighVar   ->put_rec(&vndat[0],      rec);
    vcneighVar  ->put_rec(&vcndat[0],      rec);
    directorVar ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);
    cellPosVar  ->put_rec(&cellPosDat[0],rec);
    cellTypeVar ->put_rec(&typedat[0],rec);

    File.sync();
}
