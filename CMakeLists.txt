cmake_minimum_required(VERSION 3.22)

project(cellGPU LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#note: CGAL (which I will get rid of someday) needs at least c++14; some advanced gcc compilers will throw errors if you don't set the cpp standard to 17, though
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

#set a default "Release" build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

add_definitions(-DENABLE_CUDA)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)

set(CUDA_ARCH "35;50")
                #if you have different cuda-capable hardware, modify this line to get much more optimized performance. By default,
                #I have set this to work on Tesla K40s (still used at many XSEDE facilities), but add the correct codes for optimizing performance
                #on your cards


set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp -frounding-math")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakeHelp)
find_package(CGAL REQUIRED)
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

set(SOURCES
    )
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
