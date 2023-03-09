if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/forces
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
        )

add_library(model
    DelaunayCGAL.cpp
    DelaunayGPU.cpp
    Simple2DActiveCell.cpp
    Simple2DCell.cpp
    vertexModelBase.cpp
    vertexQuadraticEnergy.cpp
    vertexQuadraticEnergyWithTension.cpp
    voronoiModelBase.cpp
    voronoiQuadraticEnergy.cpp
    voronoiQuadraticEnergyWithTension.cpp
    )

cuda_add_library(modelGPU
    DelaunayGPU.cu
    Simple2DCell.cu
    vertexModelBase.cu
    vertexQuadraticEnergy.cu
    vertexQuadraticEnergyWithTension.cu
    voronoiModelBase.cu
    voronoiQuadraticEnergy.cu
    voronoiQuadraticEnergyWithTension.cu
    )

