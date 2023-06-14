#include "logEquilibrationStateWriter.h"

logEquilibrationStateWriter::logEquilibrationStateWriter(double exponent)
    {
    logSpaceParameters.x = exponent;
    };

void logEquilibrationStateWriter::addDatabase(shared_ptr<BaseDatabase> db, int firstFrameToSave)
    {
    databases.push_back(db);
    shared_ptr<logSpacedIntegers> lsi = make_shared<logSpacedIntegers>(0,logSpaceParameters.x);
    logSpaces.push_back(lsi);
    saveOffsets.push_back(firstFrameToSave);
    nextFrames.push_back(firstFrameToSave);
    }

void logEquilibrationStateWriter::identifyNextFrame()
    {
    nextFrameToSave = LLONG_MAX;
    for (int ii = 0; ii < databases.size(); ++ii)
        {
        if(nextFrames[ii] < nextFrameToSave)
            nextFrameToSave = nextFrames[ii];
        }
    };

void logEquilibrationStateWriter::writeState(STATE c, long long int frame)
    {
    for (int ii = 0; ii < databases.size(); ++ii)
        {
        if(frame == nextFrames[ii])
            {
//            cout << frame << "\t" << ii << endl;
            databases[ii]->writeState(c);
            logSpaces[ii]->update();
            nextFrames[ii] = saveOffsets[ii]+logSpaces[ii]->nextSave;
//            cout << frame << "\t" << ii << "\t"<<nextFrames[ii] << endl;
            }
        }
    identifyNextFrame();
    }

