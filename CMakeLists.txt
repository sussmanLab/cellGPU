cmake_minimum_required(VERSION 3.0.0)

project(cellGPU)

add_definitions(-DENABLE_CUDA)

find_package(OpenMP REQUIRED)
set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr
                                        -arch=sm_30
                                        -gencode=arch=compute_30,code=sm_30
                                        -gencode=arch=compute_50,code=sm_50
                                        -gencode=arch=compute_52,code=sm_52
                                        -Wno-deprecated-gpu-targets
                                        ")
                                        #")#-gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets ")#-Xptxas -O3 -Xcompiler -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

find_package(CGAL REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakeHelp)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/databases
            $ENV{PATH}$

   )

add_subdirectory(src)

set(myLibs
    model
    modelGPU
    updaters
    updatersGPU
    analysis
    database
    simulation
    utility
    utilityGPU
    )

link_directories(${CUDA_LIBRARY_DIRS} $ENV{LD_LIBRARY_PATH})

set(SOURCES )
set(HEADERS
    )

# list the names of cpp files corresponding to linked executables you'd like...
foreach(ARG 
        voronoi
        Vertex
        )
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out"
    ${CUDA_LIBRARIES}
    ${myLibs} 
    CGAL::CGAL
    netcdf
    netcdf_c++
    OpenMP::OpenMP_CXX
    )
endforeach()
