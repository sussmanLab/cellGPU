#include "vectorValueDatabase.h"
#include "debuggingHelp.h"

valueVectorDatabase::valueVectorDatabase(std::string _filename, unsigned long vectorSize, fileMode::Enum _accessMode) : baseHDF5Database(_filename,_accessMode)
    {
    objectName = "valueVectorDatabase";
    maximumVectorSize = vectorSize;
    valueVector.resize(1);
    dataVector.resize(maximumVectorSize);
    logMessage(logger::verbose, "valueVectorDatabase initialized");
    if(_accessMode == fileMode::replace)
        {
        registerDatasets();
        };
    if(_accessMode == fileMode::readwrite)
        {
        if (currentNumberOfRecords() ==0)
            registerDatasets();
        };
    };

unsigned long valueVectorDatabase::currentNumberOfRecords()
    {
    return getDatasetDimensions("value");
    };

void valueVectorDatabase::registerDatasets()
    {
    registerExtendableDataset<double>("value",1);
    registerExtendableDataset<double>("vector",maximumVectorSize);
    };

void valueVectorDatabase::writeState(double val, std::vector<double> &data)
    {
    logMessage(logger::verbose, "valueVectorDatabase state saved");
    extendDataset("vector", data);
    valueVector[0] = val;
    extendDataset("value", valueVector);
    };

void valueVectorDatabase::readState(int record)
    {
    readDataset("value",valueVector,record);
    readDataset("vector",dataVector,record);
    };
