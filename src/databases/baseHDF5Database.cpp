#include "baseHDF5Database.h"
#include "H5Dpublic.h"
#include "H5Ppublic.h"
#include "debuggingHelp.h"
#include "fileManipulation.h"

template<> hid_t baseHDF5Database::getDatatypeFor<int>() {return H5T_NATIVE_INT;};
template<> hid_t baseHDF5Database::getDatatypeFor<float>() {return H5T_NATIVE_FLOAT;};
template<> hid_t baseHDF5Database::getDatatypeFor<double>() {return H5T_NATIVE_DOUBLE;};

/*!
a wrapper for memory spaces, using RAII to auto-handle destruction
*/ 
class h5memorySpace
    {
    public:
        h5memorySpace(int rank, const hsize_t *dims, const hsize_t *maxdims)
            {
            internalId = H5Screate_simple(rank, dims, maxdims);
            if(internalId <0)
                ERRORERROR("hdf5 cannot make a simple memory space");
            }
        ~h5memorySpace()
            {
            if(internalId >= 0)
                H5Sclose(internalId);
            }
        hid_t internalId;
    };

/*!
a wrapper for data spaces, using RAII to auto-handle destruction
*/ 
class h5dataSpaceCreate
    {
    public:
        h5dataSpaceCreate(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
            {
            internalId = H5Dcreate(loc_id, name,type_id,space_id,lcpl_id,dcpl_id,dapl_id);
            if(internalId <0)
                ERRORERROR("hdf5 cannot make a simple data space");
            }
        ~h5dataSpaceCreate()
            {
            if(internalId >= 0)
                H5Dclose(internalId);
            }
        hid_t internalId;
    };
/*!
a wrapper for data spaces, using RAII to auto-handle destruction
*/ 
class h5dataSpaceOpen
    {
    public:
        h5dataSpaceOpen(hid_t loc_id, const char *name, hid_t dapl_id)
            {
            internalId = H5Dopen(loc_id, name, dapl_id);
            if(internalId <0)
                ERRORERROR("hdf5 cannot open a simple data space");
            }
        ~h5dataSpaceOpen()
            {
            if(internalId >= 0)
                H5Dclose(internalId);
            }
        hid_t internalId;
    };

/*!
a wrapper for property lists, using RAII to auto-handle destruction
*/ 
class h5propertyList
    {
    public:
        h5propertyList(hid_t cls_id)
            {
            internalId = H5Pcreate(cls_id);
            if(internalId <0)
                ERRORERROR("hdf5 cannot make a property list");
            }
        ~h5propertyList()
            {
            if(internalId >= 0)
                H5Pclose(internalId);
            }
        hid_t internalId;
    };

baseHDF5Database::baseHDF5Database(std::string _filename, fileMode::Enum  _accessMode)
    {
    objectName = "baseHDF5Database";
    filename = _filename;
    mode = _accessMode;
    if(_accessMode == fileMode::readonly && !fileExists(filename))
        ERRORERROR("trying to read a file that does not exist\n");
    
    // Create/replace hdf5 file, and currently fail if you want readwrite mode
    if(mode == fileMode::readonly)
        {
        hdf5File = H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
        }
    else if(mode == fileMode::readwrite)
        {
        createDirectoriesOnPath(filename);
        if(fileExists(filename))
            hdf5File = H5Fopen(filename.c_str(),H5F_ACC_RDWR,H5P_DEFAULT);
        else
            hdf5File = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        }
    else
        {
        createDirectoriesOnPath(filename);
        hdf5File = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        };

    };

baseHDF5Database::~baseHDF5Database()
    {
    H5Fclose(hdf5File);
    };

template<typename T>
void baseHDF5Database::addHeaderData(std::string name, const std::vector<T> &data)
    {
    if(mode == fileMode::readonly)
        ERRORERROR("don't write on a readonly file");
    const hsize_t ndims=1;
    const hsize_t ncols=data.size();
    hsize_t dims[ndims] = {ncols};
    hsize_t max_dims[ndims] = {ncols};
    h5memorySpace fileSpace(ndims, dims, max_dims);
    h5propertyList plist(H5P_DATASET_CREATE);
    H5Pset_layout(plist.internalId, H5D_COMPACT);

    h5dataSpaceCreate dataset(hdf5File,name.c_str(),getDatatypeFor<T>(),fileSpace.internalId,H5P_DEFAULT,plist.internalId,H5P_DEFAULT);

    H5Dwrite(dataset.internalId, getDatatypeFor<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Fflush(hdf5File, H5F_SCOPE_GLOBAL);
    };

unsigned long baseHDF5Database::getDatasetDimensions(std::string name)
    {
    h5dataSpaceOpen dataset(hdf5File,name.c_str(),H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset.internalId);
    //get the current dimensions of the dataset
    hsize_t dims[2];
    herr_t ndims = H5Sget_simple_extent_dims(dataspace,dims,NULL);
    if(ndims<0)
        ERRORERROR("failed to get dimensions\n");
    H5Dclose(dataspace);
    return dims[0];
    };

template<typename T>
void baseHDF5Database::extendDataset(std::string name,std::vector<T> &data)
    {
    if(mode == fileMode::readonly)
        ERRORERROR("don't write on a readonly file");
    h5dataSpaceOpen dataset(hdf5File,name.c_str(),H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset.internalId);
    //get the current dimensions of the dataset
    hsize_t dims[2];
    herr_t ndims = H5Sget_simple_extent_dims(dataspace,dims,NULL);
    if(ndims<0)
        ERRORERROR("incorrect dimensions on extending dataset\n");
    if(data.size()!= dims[1])
        ERRORERROR("trying to write a vector of the wrong size for the dataset");

    //allocate space to extend the dataset
    hsize_t newDimensions[2] = {dims[0]+1,dims[1]};
    H5Dset_extent(dataset.internalId,newDimensions);

    hsize_t newMemoryDimensions[2] = {1,dims[1]};
    h5memorySpace memorySpace(ndims,newMemoryDimensions,NULL);
    dataspace = H5Dget_space(dataset.internalId);
    hsize_t offset[2] = {dims[0],0};
    hsize_t count[2] = {1,dims[1]};
    H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);
    H5Dwrite(dataset.internalId,getDatatypeFor<T>(),memorySpace.internalId,dataspace,H5P_DEFAULT,data.data());
    H5Sclose(dataspace);
    H5Fflush(hdf5File, H5F_SCOPE_GLOBAL);
    };

template<typename T>
void baseHDF5Database::registerExtendableDataset(std::string name, int maximimumSizePerRecord)
    {
    if(mode == fileMode::readonly)
        ERRORERROR("don't write on a readonly file");
    extendableDatasetNames.insert(name);

    const hsize_t ndims=2;
    const hsize_t ncols=maximimumSizePerRecord;

    //create a file space as a 2D type of data with potentially unlimited rows
    hsize_t dims[ndims] = {0, ncols};
    hsize_t max_dims[ndims] = {H5S_UNLIMITED, ncols};
    h5memorySpace fileSpace(ndims, dims, max_dims);

    // Create a dataset creation property list
    // The layout of the dataset have to be chunked when using unlimited dimensions
    h5propertyList propertyList(H5P_DATASET_CREATE);
    H5Pset_layout(propertyList.internalId, H5D_CHUNKED);
    //we intend to write one record at a time, so set the chunk size to a complete row
    hsize_t chunk_dims[ndims] = {1, ncols};
    H5Pset_chunk(propertyList.internalId, ndims, chunk_dims);

    // Create the dataset
    h5dataSpaceCreate dataset(hdf5File, name.c_str(), getDatatypeFor<T>(), fileSpace.internalId, H5P_DEFAULT, propertyList.internalId, H5P_DEFAULT);
    };

template<typename T>
void baseHDF5Database::readDataset(std::string name,std::vector<T> &data, int record)
    {
    h5dataSpaceOpen dataset(hdf5File,name.c_str(),H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset.internalId);
    //get the current dimensions of the dataset
    hsize_t dims[2];
    int ndims = H5Sget_simple_extent_dims(dataspace,dims,NULL);
    if(ndims!=2)
        ERRORERROR("reading a dataset that isn't two dimensional\n");
    if(data.size()!= dims[1])
        ERRORERROR("trying to read a vector of the wrong size for the dataset");

    hsize_t rowIndex = record;
    if(record < 0)
        rowIndex = dims[0]-1;
    std::cerr << record << " " <<dims[0] << " " << dims[1] << " " << rowIndex << "\n";
    if(rowIndex >= dims[0])
        ERRORERROR("Trying to read past the end of the dataset\n");

    hsize_t offset[2] = {rowIndex,0};
    hsize_t count[2] = {1,dims[1]};
    H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL);

    h5memorySpace memorySpace(ndims,count,NULL);

    H5Dread(dataset.internalId,getDatatypeFor<T>(),memorySpace.internalId,dataspace, H5P_DEFAULT, data.data());

    H5Dclose(dataspace);
    };

void baseHDF5Database::readTest(int record)
    {
    std::vector<int> readInts(20);
    std::vector<double> readDoubles(5);
    readDataset("/extendableDoubles", readDoubles);
    for (unsigned long ii = 0; ii < readDoubles.size();++ii)
        std::cerr << readDoubles[ii] <<"\t";
    std::cerr << std::endl;
    readDataset("/extendableDoubles", readDoubles,record);
    for (unsigned long ii = 0; ii < readDoubles.size();++ii)
        std::cerr << readDoubles[ii] <<"\t";
    std::cerr << std::endl;
    readDataset("/extendableIntegerRecord", readInts,record);
    for (unsigned long ii = 0; ii < readInts.size();++ii)
        std::cerr << readInts[ii] <<"\t";
    std::cerr << std::endl;
    };

void baseHDF5Database::writeTest()
    {
    std::vector<int> dataToSave{123,213,312,124444,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    addHeaderData("/integersOfInterest",dataToSave);

    registerExtendableDataset<double>("/extendableDoubles", 5);
    std::vector<double> ds3{0.1123123132, 0.01, -12.0, 4.4, 5.5};
    extendDataset("/extendableDoubles", ds3);
    ds3[2]=123321.123;
    extendDataset("/extendableDoubles", ds3);
    ds3[0]=1.123;
    extendDataset("/extendableDoubles", ds3);
    extendDataset("/extendableDoubles", ds3);
    extendDataset("/extendableDoubles", ds3);
    extendDataset("/extendableDoubles", ds3);

    registerExtendableDataset<int>("/extendableIntegerRecord", 20);
    std::vector<int> intSet(20,0);
    extendDataset("/extendableIntegerRecord",intSet);
    intSet[0] = 1;
    extendDataset("/extendableIntegerRecord",intSet);
    intSet[5] = 5;
    extendDataset("/extendableIntegerRecord",intSet);
    intSet[13] = 123;
    extendDataset("/extendableIntegerRecord",intSet);
    intSet[8] = 9;
    extendDataset("/extendableIntegerRecord",intSet);
    intSet[10] = 10;
    extendDataset("/extendableIntegerRecord",intSet);
    };


template void baseHDF5Database::addHeaderData<int>(std::string,const std::vector<int>&);
template void baseHDF5Database::addHeaderData<float>(std::string, const std::vector<float>&);
template void baseHDF5Database::addHeaderData<double>(std::string,const std::vector<double>&);

template void baseHDF5Database::registerExtendableDataset<int>(std::string,int);
template void baseHDF5Database::registerExtendableDataset<float>(std::string,int);
template void baseHDF5Database::registerExtendableDataset<double>(std::string,int);

template void baseHDF5Database::extendDataset<int>(std::string, std::vector<int>&);
template void baseHDF5Database::extendDataset<float>(std::string, std::vector<float>&);
template void baseHDF5Database::extendDataset<double>(std::string, std::vector<double>&);

template void baseHDF5Database::readDataset<int>(std::string,std::vector<int> &, int);
template void baseHDF5Database::readDataset<float>(std::string,std::vector<float> &, int);
template void baseHDF5Database::readDataset<double>(std::string,std::vector<double> &, int);
