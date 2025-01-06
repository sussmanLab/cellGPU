#ifndef baseDatabase_h
#define baseDatabase_h

#include "std_include.h"
#include "Simple2DCell.h"

/*! \file BaseDatabase.h */
//! A base class defining the operations of a database save and read scheme

class BaseDatabase
    {
    protected:
        typedef shared_ptr<Simple2DCell> STATE;
    public:
        //! Base constructure takes a bland filename in readonly mode
        BaseDatabase(string fn="temp.txt", int mode=-1):filename(fn), Mode(mode),Records(0){};
        //!The name of the file
        string filename;
        //!The desired mode (integer representation of replace, new, write, readonly, etc)
        const int Mode;
        //!The number of saved records in the database
        int Records;

        //!Write the current state; if the default value of rec=-1 is used, add a new entry
        virtual void writeState(STATE c, double time = -1.0, int rec = -1){};
        //Read the rec state of the database. If geometry = true, call computeGeomerty routines (instead of just reading in the d.o.f.s)
        virtual void readState(STATE c, int rec, bool geometry = true){};

    };
#endif
