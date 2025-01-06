#include "simpleVoronoiDatabase.h"
#include "baseHDF5Database.h"
#include "debuggingHelp.h"

simpleVoronoiDatabase::simpleVoronoiDatabase(int np, string fn,fileMode::Enum _mode) 
    : baseHDF5Database(fn,_mode)
    {
    objectName = "simpleVoronoiDatabase";
    N = np;
    timeVector.resize(1);
    boxVector.resize(4);
    intVector.resize(N);
    doubleVector.resize(N);
    coordinateVector.resize(2*N);

    if(mode == fileMode::replace)
        {
        registerDatasets();
        };
    if(mode == fileMode::readwrite)
        {
        if (currentNumberOfRecords() ==0)
            registerDatasets();
        };
    logMessage(logger::verbose, "modelDatabase initialized");
    }

void simpleVoronoiDatabase::registerDatasets()
    {
    registerExtendableDataset<double>("time",1);
    registerExtendableDataset<double>("boxMatrix",4);

    registerExtendableDataset<int>("type",N);

    registerExtendableDataset<double>("position", 2*N);
    registerExtendableDataset<double>("velocity", 2*N);
    // registerExtendableDataset<double>("additionalData", 2*N);
    }

void simpleVoronoiDatabase::writeState(STATE c, double time, int rec)
    {
    if(rec >= 0)
        ERRORERROR("overwriting specific records not implemented at the moment");
    shared_ptr<voronoiModelBase> s = dynamic_pointer_cast<voronoiModelBase>(c);
    if (time < 0) time = s->currentTime;
    //time
    timeVector[0] = time;
    extendDataset("time", timeVector);

    //boxMatrix
    double x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxVector[0]=x11;
    boxVector[1]=x12;
    boxVector[2]=x21;
    boxVector[3]=x22;
    extendDataset("boxMatrix",boxVector);

    //type
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        intVector[ii] = h_ct.data[pidx];
        }
    extendDataset("type",intVector);

    //position
    ArrayHandle<double2> h_p(s->returnPositions(),access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        coordinateVector[2*ii] = h_p.data[pidx].x;
        coordinateVector[2*ii+1] = h_p.data[pidx].y;
        }
    extendDataset("position",coordinateVector); 

    //velocity
    ArrayHandle<double2> h_v(s->returnVelocities(),access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        coordinateVector[2*ii] = h_v.data[pidx].x;
        coordinateVector[2*ii+1] = h_v.data[pidx].y;
        }
    extendDataset("velocity",coordinateVector); 

    }

void simpleVoronoiDatabase::readState(STATE c, int rec, bool geometry)
    {
    shared_ptr<voronoiModelBase> t = dynamic_pointer_cast<voronoiModelBase>(c);

    readDataset("time",timeVector,rec);

    readDataset("boxMatrix",boxVector,rec);
    t->Box->setGeneral(boxVector[0],boxVector[1],boxVector[2],boxVector[3]);

    readDataset("type",intVector,rec);
    ArrayHandle<int> h_ct(t->cellType,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < N; ++idx)
        {
        h_ct.data[idx]=intVector[idx];;
        };

    readDataset("position",coordinateVector,rec);
    ArrayHandle<double2> h_p(t->cellPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < N; ++idx)
        {
        h_p.data[idx].x = coordinateVector[(2*idx)];
        h_p.data[idx].y = coordinateVector[(2*idx)+1];
        };

    readDataset("velocity",coordinateVector,rec);
    ArrayHandle<double2> h_v(t->returnVelocities(),access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < N; ++idx)
        {
        h_v.data[idx].x = coordinateVector[(2*idx)];
        h_v.data[idx].y = coordinateVector[(2*idx)+1];
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

unsigned long simpleVoronoiDatabase::currentNumberOfRecords()
    {
    return getDatasetDimensions("time");
    }
