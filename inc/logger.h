#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <string>

/*!
logger is a singleton that can receive messages from loggable objects.
It has a "level" of message that will be printed to screen, messages with an
enum level below that will be printed and those with an enum level above will be
ignored.
*/
class logger {
public:
    //! Log levels
    enum logLevel
        {
        none = 0,
        error = 1,
        warning = 2,
        info = 3,
        verbose = 4,
        debug = 5
        };

    //!Get the singleton instance
    static logger& instance()
        {
        static logger instance(info); // Set the default log level here
        return instance;
        }

    //! Set the log level to an enum value
    void setLogLevel(logLevel level)
        {
        logLevel = level;
        }

    //! Get the log level
    logLevel getLogLevel() const 
        {
        return logLevel;
        }

    //! Log a message with a given level and source
    void log(logLevel level, const std::string& source, const std::string& message)
        {
        if (level <= logLevel)
            {
            std::cout << "[" << logLevelToString(level) << "] (" 
                    << source << "): " << message << std::endl;
            }
        }

private:
    logger(logLevel level) : logLevel(level) {} //!Private constructor
    logger(const logger&) = delete; //! Prevent copying
    logger& operator=(const logger&) = delete; //! Prevent assignment
    logLevel logLevel;


    //! Helper function to convert logLevel to string (for output)
    std::string logLevelToString(enum logLevel level) const {
        switch (level) {
            case none:    return "NONE";
            case error:   return "ERROR";
            case warning: return "WARNING";
            case info:    return "INFO";
            case verbose: return "VERBOSE";
            case debug:   return "DEBUG";
            default:      return "UNKNOWN";
        }
    }
};
#endif
