#include "DatabaseTextVoronoi.h"
/*! \file DatabaseTextVoronoi.cpp */

DatabaseTextVoronoi::DatabaseTextVoronoi(string fn, int mode)
    : BaseDatabase(fn,mode)
    {
    switch(Mode)
        {
        case -1:
            inputFile.open(filename);
            break;
        case 0:
            outputFile.open(filename);
            break;
        case 1:
            outputFile.open(filename, ios::app);
            break;
        default:
            ;
        };
    };

void DatabaseTextVoronoi::writeState(STATE s, double time, int rec)
    {
    if (rec != -1)
        {
        printf("writing to the middle of text files not supported\n");
        throw std::exception();
        };
        
    int N = s->getNumberOfDegreesOfFreedom();
//    printf("saving %i cells\n",N);
    if (time < 0) time = s->currentTime;
    double x11,x12,x21,x22;
    s->returnBox().getBoxDims(x11,x12,x21,x22);

    outputFile << "N, time, box:\n";
    outputFile << N << "\t" << time <<"\t" <<x11 <<"\t" <<x12<<"\t" <<x21<<"\t" <<x22 <<"\n";

    ArrayHandle<double2> h_pos(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        outputFile <<h_pos.data[pidx].x <<"\t"<<h_pos.data[pidx].y <<"\t" <<h_ct.data[pidx] <<"\n";
        };
    };

void DatabaseTextVoronoi::readState(STATE s, int rec, bool geometry)
    {
    printf("Reading from a text database currently not supported\n");
    throw std::exception();
    };
