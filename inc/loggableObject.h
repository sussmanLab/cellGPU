#ifndef LOGGABLEOBJECT_H
#define LOGGABLEOBJECT_H

#include "logger.h"

//! a loggableObject has a name, and can pass messages to the singleton logger with different levels
class loggableObject {
public:
    loggableObject(const std::string& _name) : objectName(_name)
        {
        }

    void logMessage(enum logger::logLevel level, const std::string& message) 
        {
        logger::instance().log(level, objectName, message);
        }

    std::string objectName;
};

#endif
