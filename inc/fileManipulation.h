#ifndef FILEMANIPULATION_H
#define FILEMANIPULATION_H

#include <fstream>
#include <filesystem>

//!A utility function for checking if a file exists
inline bool fileExists(const std::string& name)
    {
    std::ifstream f(name.c_str());
    return f.good();
    }

//!A utility: when handed filename = "path/to/file.txt", creates directories if they don't already exist
inline void createDirectoriesOnPath(const std::string &filename)
    {
    std::filesystem::path filePath(filename);
    std::filesystem::path parentPath = filePath.parent_path();
    if(!parentPath.empty())
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    }

#endif
