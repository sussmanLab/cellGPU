#ifndef dbTextVoronoi_h
#define dbTextVoronoi_h

#include "BaseDatabase.h"
/*! \file DatabaseTextVoronoi.h */
//! A simple text-based output for voronoi models...only supports sequential writing of frames

/*!
The text database format is as follows. For each frame, a line is written with N (either vertices or cells), the time, and the four box dimensions.
There is then a line for each cell/vertex, with position and cell type (or position and vertex connections).
All of the above data is just tab delimited. Writing to the middle of a file is not supported
*/
class DatabaseTextVoronoi : public BaseDatabase
    {
    public:
        //!constructor prepares the stream class
        DatabaseTextVoronoi(string fn = "temp.txt", int mode = -1);
        //!Write the current state; if the default value of rec=-1 is used, add a new entry
        virtual void writeState(STATE c, double time = -1.0, int rec = -1);
        //Read the rec state of the database. If geometry = true, call computeGeomerty routines (instead of just reading in the d.o.f.s)
        virtual void readState(STATE c, int rec, bool geometry = true);
    
    protected:
        ofstream outputFile;   
        ifstream inputFile;   
    };

#endif
