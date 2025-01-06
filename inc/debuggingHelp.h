#ifndef DEBUGGINGHELP_H
#define DEBUGGINGHELP_H
#include <iostream>

//!report the line and file we've gotten to
inline static void debugCodeHelper(const char *file, const int line)
    {
    std::cerr << "Reached file " << file<<" at line " << line <<"\n";
    };

//!Report somewhere that code needs to be written
static void unwrittenCode(const char *message, const char *file, int line)
    {
    std::cerr << "Code unwritten (file "<<file<<"; line "<< line << "\nMessage: " << message <<"\n";
    throw std::exception();
    };

//!display an error message (plus file and line information), then throw an exception
static void errorFound(const char *message, const char *file, int line)
    {
    std::cerr << "Error identified (file "<<file<<"; line "<< line << "\nMessage: " << message <<"\n";
    throw std::exception();
    };

//!spot-checking of code for debugging
#define DEBUGCODEHELPER (debugCodeHelper(__FILE__,__LINE__))

//!A macro to say code needs to be written
#define UNWRITTENCODE(message) (unwrittenCode(message,__FILE__,__LINE__))
//!a macro to say something is wrong!
#define ERRORERROR(message) (errorFound(message,__FILE__,__LINE__))

#endif
