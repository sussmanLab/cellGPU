#ifndef profiler_H
#define profiler_H

#include <chrono>
#include <string>
#include <iostream>
using namespace std;
class profiler
    {
    public:
        profiler(){functionCalls = 0; timeTaken = 0;};
        profiler(string profilerName) : name(profilerName) {functionCalls = 0; timeTaken = 0;};

        void start()
            {
            startTime = chrono::high_resolution_clock::now();
            };
        void end()
            {
            endTime = chrono::high_resolution_clock::now();
            chrono::duration<double> difference = endTime-startTime;
            timeTaken += difference.count();
            functionCalls +=1;
            };

        double timing()
            {
            if(functionCalls>0)
                return timeTaken/functionCalls;
            else
                return 0;
            };

        void print()
            {
            cout << "profiler \"" << name << "\" took an average of " << timing() << " per call over " << functionCalls << " calls...total time = "<<timing()*functionCalls << endl;
            }

        void setName(string _name){name=_name;};
        chrono::time_point<chrono::high_resolution_clock>  startTime;
        chrono::time_point<chrono::high_resolution_clock>  endTime;
        int functionCalls;
        double timeTaken;
        string name;
    };
#endif
