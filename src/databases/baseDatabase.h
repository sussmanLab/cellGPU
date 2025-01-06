#ifndef BASEDATABASE_H
#define BASEDATABASE_H
#include <string>
#include "loggableObject.h"

//!A structure for declaring how we want to access data (read, write, overwrite?)
struct fileMode
    {
    //!An enumeration of possibilities
    enum Enum
        {
        readonly,       //!< we just want to read
        readwrite,  //!< we intend to both read and write
        replace   //!< we will completely overwrite all of the data
        };
    };

//!  baseDatabaseInformation just has a filename and an access mode
class baseDatabaseInformation : public loggableObject
    {
    public:
        baseDatabaseInformation() : loggableObject("database") {};
        std::string filename;
        fileMode::Enum mode;
        //!The number of saved records in the database
        int Records;
    };
#endif
