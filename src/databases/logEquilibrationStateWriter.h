#ifndef logEqStateWriter_H
#define logEqStateWriter_H

#include "BaseDatabase.h"
#include "analysisPackage.h"

//! Handles the logic of saving log-spaced data files that might begin at different offset times
class logEquilibrationStateWriter
    {
    private:
        typedef shared_ptr<Simple2DCell> STATE;

    public:
        logEquilibrationStateWriter(double exponent = 0.1);


        void addDatabase(shared_ptr<BaseDatabase> db, int firstFrameToSave);
        void identifyNextFrame();
        void writeState(STATE c, long long int frame);

        vector<shared_ptr<BaseDatabase>> databases;
        vector<int> saveOffsets;
        vector<shared_ptr<logSpacedIntegers>> logSpaces;
        vector<long long int> nextFrames;

        long long int nextFrameToSave; 

        double2 logSpaceParameters;

    };
#endif
