#ifndef mprofiler_H
#define mprofiler_H

#include <chrono>
#include <string>
#include <iostream>
#include <map>
using namespace std;
class multiProfiler
    {
    public:
        multiProfiler(){numberOfProfiles = 0;};

        void addName(string name)
            {
            map<string,int>::iterator it = processMap.find(name);
            if(it !=processMap.end())
                {
                processMap.insert(std::pair<string,int>(name,numberOfProfiles) );
                functionCalls.push_back(0);
                timeTaken.push_back(0);
                numberOfProfiles += 1;
                }
            }
        void start(string name)
            {
            map<string,int>::iterator it = processMap.find(name);

            if(it !=processMap.end())
                startTimes[(*it).second] = chrono::high_resolution_clock::now();
            else
                {
                processMap.insert(std::pair<string,int>(name,numberOfProfiles) );
                functionCalls.push_back(0);
                timeTaken.push_back(0);
                endTimes.push_back(chrono::high_resolution_clock::now());
                startTimes.push_back(chrono::high_resolution_clock::now());
                numberOfProfiles += 1;
                }
            };


        void end(string name)
            {
            map<string,int>::iterator it = processMap.find(name);
            int idx = (*it).second;
            endTimes[idx] = chrono::high_resolution_clock::now();
            chrono::duration<double> difference = endTimes[idx]-startTimes[idx];
            timeTaken[idx] += difference.count();
            functionCalls[idx] +=1;
            };

        double timing(string name)
            {
            map<string,int>::iterator it = processMap.find(name);
            double ans =0;
            if(it != processMap.end() && functionCalls[(*it).second] >0 )
                ans = timeTaken[(*it).second]/functionCalls[(*it).second] ;
            return ans;
            };

        void print()
            {
            for (map<string,int>::iterator it = processMap.begin(); it != processMap.end(); ++it)
                {
                cout << "process "<<(*it).first << " averaged\t " << timing((*it).first)
                     << " \t over " << functionCalls[(*it).second] << " function calls" << endl;
                }
            }

        int numberOfProfiles;
        vector<chrono::time_point<chrono::high_resolution_clock> >  startTimes;
        vector<chrono::time_point<chrono::high_resolution_clock> > endTimes;
        vector<int> functionCalls;
        vector<double> timeTaken;
        map<string,int> processMap;
    };
#endif
