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

add_library(updaters
    velocityVerlet.cpp
    brownianParticleDynamics.cpp
    EnergyMinimizerFIRE2D.cpp
    MullerPlatheShear.cpp
    NoseHooverChainNVT.cpp
    selfPropelledAligningParticleDynamics.cpp
    selfPropelledCellVertexDynamics.cpp
    selfPropelledParticleDynamics.cpp
    selfPropelledVicsekAligningParticleDynamics.cpp
    setTotalLinearMomentum.cpp
    langevinDynamics.cpp
    VSSRNEMD.cpp
    )
cuda_add_library(updatersGPU
    EnergyMinimizerFIRE2D.cu
    NoseHooverChainNVT.cu
    brownianParticleDynamics.cu
    selfPropelledAligningParticleDynamics.cu
    selfPropelledCellVertexDynamics.cu
    selfPropelledParticleDynamics.cu
    selfPropelledVicsekAligningParticleDynamics.cu
    setTotalLinearMomentum.cu
    simpleEquationOfMotion.cu
    langevinDynamics.cu
    )
