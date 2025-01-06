add_library(databases
    baseHDF5Database.cpp
    logEquilibrationStateWriter.cpp
    # DatabaseNetCDFAVM.cpp
    # DatabaseNetCDFSPV.cpp
    databaseTextVoronoi.cpp
    vectorValueDatabase.cpp
    simpleVoronoiDatabase.cpp
    simpleVertexDatabase.cpp
    )

target_include_directories(databases PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(databases PUBLIC ${HDF5_LIBRARIES})
